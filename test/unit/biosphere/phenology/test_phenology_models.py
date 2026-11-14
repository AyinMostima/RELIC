"""
Unit Tests for Concrete Phenology Models

This test suite validates the concrete implementations of phenological models
including StaticPhenology, GDDPhenology, GSIPhenology, and CropPhenology.
Tests focus on mathematical accuracy, scientific validity, and integration
compliance with the RELIC framework.

Test Coverage:
    - Model initialization and parameter validation
    - Phenological state computation accuracy
    - Seasonal transition logic
    - Biomass flux calculations
    - Physical constraint enforcement
    - Integration with FunctorBase and BiologicalProcess
    - Edge cases and boundary conditions

Scientific References:
    - Jolly, W.M., et al. (2005). GSI model formulation. Global Change Biology, 11(4), 619-632
    - White, M.A., et al. (1997). Continental phenology modeling. Global Biogeochemical Cycles, 11(2), 217-234
    - McMaster, G.S. & Wilhelm, W.W. (1997). GDD interpretation. Agricultural and Forest Meteorology, 87(4), 291-300
"""

import inspect

import pytest
import numpy as np
import torch
import logging

# Import phenology models
from relic.scientific.biosphere.phenology import (
    StaticPhenology, GDDPhenology, GSIPhenology, CropPhenology,
    PhenologyParameters, PhenologyParameterSets, PhenologyPhase, PhenologyOutput
)
from relic.scientific.biosphere.phenology.utils import (
    compute_daylength
)
from relic.utils.physical_constants import get_physical_constant
from relic.parameters import get_global_manager
from relic.core.functor import ExecutionError
from relic.core.state import UpdateMode

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Test fixtures
@pytest.fixture
def static_phenology_config():
    """Configuration for static phenology model"""
    return {
        'name': 'static_test',
        'phenology_type': 'static',
        'leaf_onset_doy': 120,  # May 1
        'leaf_offset_doy': 280,  # October 7
        'latitude': 45.0,
        'max_lai': 5.0,
        'leaf_turnover_rate': 0.3,
        'froot_turnover_rate': 1.2
    }

@pytest.fixture
def gdd_phenology_config():
    """Configuration for GDD phenology model"""
    return {
        'name': 'gdd_test',
        'phenology_type': 'gdd',
        'gdd_base_temp': -5.0,
        'gdd_threshold_start': 200.0,
        'gdd_threshold_end': 1200.0,
        'latitude': 50.0,
        'max_lai': 6.0,
        'leaf_turnover_rate': 0.25,
        'froot_turnover_rate': 1.0
    }

@pytest.fixture 
def gsi_phenology_config():
    """Configuration for GSI phenology model"""
    return {
        'name': 'gsi_test',
        'phenology_type': 'gsi',
        'gsi_tmin_threshold': -2.0,
        'gsi_tmin_optimal': 5.0,
        'gsi_vpd_min': 900.0,
        'gsi_vpd_max': 4100.0,
        'gsi_daylen_threshold': 39300.0,  # 10.9 hours in seconds
        'gsi_onset_threshold': 0.5,
        'gsi_offset_threshold': 0.2,
        'gsi_memory_days': 7,
        'latitude': 45.0,
        'max_lai': 6.5,
        'leaf_turnover_rate': 0.2,
        'froot_turnover_rate': 0.8
    }


GSI_TEMPORAL_STATE_KEYS = (
    "phenology_gsi_history",
    "phenology_gsi_history_index",
    "phenology_gsi_history_count",
    "phenology_gsi_in_growing_season",
)


def _gsi_temporal_state(
    model: GSIPhenology,
    *,
    history: torch.Tensor | None = None,
    index: int = 0,
    count: int = 0,
    in_growing_season: bool = False,
) -> dict[str, torch.Tensor]:
    """Build canonical StateManager-owned GSI temporal inputs."""
    if history is None:
        history = torch.zeros(model.memory_days, dtype=torch.float32)
    return {
        "phenology_gsi_history": history,
        "phenology_gsi_history_index": torch.tensor(float(index)),
        "phenology_gsi_history_count": torch.tensor(float(count)),
        "phenology_gsi_in_growing_season": torch.tensor(
            float(in_growing_season)
        ),
    }


CROP_TEMPORAL_STATE_UNITS = {
    "phenology_crop_accumulated_phu": "degC day",
    "phenology_crop_is_sown": "dimensionless",
    "phenology_crop_is_harvested": "dimensionless",
    "phenology_crop_vernalization_days": "day",
}
CROP_TEMPORAL_STATE_KEYS = tuple(CROP_TEMPORAL_STATE_UNITS)


def _crop_temporal_state(
    *,
    accumulated_phu: float = 0.0,
    is_sown: bool = False,
    is_harvested: bool = False,
    vernalization_days: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Build canonical StateManager-owned Crop temporal inputs."""
    return {
        "phenology_crop_accumulated_phu": torch.tensor(float(accumulated_phu)),
        "phenology_crop_is_sown": torch.tensor(float(is_sown)),
        "phenology_crop_is_harvested": torch.tensor(float(is_harvested)),
        "phenology_crop_vernalization_days": torch.tensor(
            float(vernalization_days)
        ),
    }


def _advance_crop(
    model: CropPhenology,
    temporal_state: dict[str, torch.Tensor],
    *,
    day_of_year: float,
    temperature_c: float,
    dt: float | None = None,
) -> tuple[PhenologyOutput, dict[str, torch.Tensor]]:
    """Advance Crop phenology from explicit canonical temporal state."""
    seconds_per_day = float(get_physical_constant("seconds_per_day"))
    if dt is None:
        dt = seconds_per_day
    result = model.compute_phenological_state(
        {
            **temporal_state,
            "day_of_year": torch.tensor([float(day_of_year)]),
            "air_temperature": torch.tensor(
                [get_physical_constant("T_FREEZE") + float(temperature_c)]
            ),
        },
        dt=dt,
    )
    next_state = {
        key: result.additional_outputs[key].detach().clone()
        for key in CROP_TEMPORAL_STATE_KEYS
    }
    return result, next_state

@pytest.fixture
def crop_phenology_config():
    """Configuration for crop phenology model"""
    values = PhenologyParameterSets.maize_crop().model_dump()
    values.update(
        crop_phu_total=500.0,
        crop_hi_max=0.45,
        crop_hi_start_fraction=0.4,
        crop_sowing_doy=120,
        crop_harvest_doy=280,
    )
    params = PhenologyParameters(**values)
    return {'config': {}, 'params': params}


@pytest.fixture
def winter_wheat_phenology_config():
    """Configuration for cross-year winter wheat phenology."""
    return {
        'config': {},
        'params': PhenologyParameterSets.winter_wheat_crop(),
    }

@pytest.fixture
def seasonal_environment_data():
    """Environmental data representing a full growing season"""
    days = torch.arange(1, 366)  # Full year
    
    # Sinusoidal temperature pattern (simplified)
    temp_mean = 10.0 + 15.0 * torch.sin(2 * np.pi * (days - 80) / 365.0)
    temp_max = temp_mean + 8.0
    temp_min = temp_mean - 8.0
    
    # VPD pattern (higher in summer)
    vpd = 1000.0 + 1500.0 * torch.sin(2 * np.pi * (days - 80) / 365.0).clamp(min=0)
    
    # Daylength variation
    latitude = 45.0  # degrees
    daylen = compute_daylength(days, latitude, calendar_type="noleap")
    
    # Simplified soil water potential
    swp = -0.01 - 0.15 * torch.sin(2 * np.pi * (days - 200) / 365.0).clamp(min=0)
    
    return {
        'day_of_year': days,
        'air_temperature': temp_mean,
        'air_temperature_max': temp_max,
        'air_temperature_min': temp_min,
        'vapor_pressure_deficit': vpd,
        'daylength': daylen,
        'soil_water_potential': swp
    }

@pytest.fixture
def standard_vegetation_state():
    """Standard vegetation state for testing"""
    batch_size = 10
    return {
        'leaf_carbon': torch.ones(batch_size) * 1.0,
        'froot_carbon': torch.ones(batch_size) * 0.5,
        'stem_carbon': torch.ones(batch_size) * 3.0,
        'leaf_nitrogen': torch.ones(batch_size) * 0.08,
        'froot_nitrogen': torch.ones(batch_size) * 0.04,
        'lai': torch.ones(batch_size) * 3.0
    }


class TestStaticPhenology:
    """Test suite for StaticPhenology model"""
    
    def test_static_phenology_initialization(self, static_phenology_config):
        """Test static phenology model initialization"""
        model = StaticPhenology(config=static_phenology_config)
        
        assert model.metadata.name == 'static_phenology'
        assert hasattr(model, 'phenology_params')
        param_manager = get_global_manager()
        expected_onset = param_manager.get('biosphere.phenology.static.leaf_onset_doy')
        expected_offset = param_manager.get('biosphere.phenology.static.leaf_offset_doy')
        assert model.phenology_params.leaf_onset_doy == expected_onset
        assert model.phenology_params.leaf_offset_doy == expected_offset
        
    def test_static_phenology_spring_onset(self, static_phenology_config):
        """Test spring onset behavior"""
        model = StaticPhenology(config=static_phenology_config)
        
        # Test before onset (day 100)
        doy_before = torch.tensor([100.0])
        temp = torch.tensor([15.0])
        
        result_before = model.compute_phenological_state(
            {
                'air_temperature_max': temp + 5.0,
                'air_temperature_min': temp - 5.0,
                "day_of_year": doy_before,
                "air_temperature": temp,
            },
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert result_before.phenology_phase == PhenologyPhase.DORMANCY
        assert result_before.growth_factor < 0.1
        
        # Test after onset (day 140)
        doy_after = torch.tensor([140.0])
        
        result_after = model.compute_phenological_state(
            {
                'air_temperature_max': temp + 5.0,
                'air_temperature_min': temp - 5.0,
                "day_of_year": doy_after,
                "air_temperature": temp,
            },
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert result_after.phenology_phase == PhenologyPhase.LEAF_EXPANSION
        assert result_after.growth_factor > 0.5
        
    def test_static_phenology_autumn_senescence(self, static_phenology_config):
        """Test autumn senescence behavior"""
        model = StaticPhenology(config=static_phenology_config)
        
        # Test during growing season (day 200)
        doy_growing = torch.tensor([200.0])
        temp = torch.tensor([20.0])
        
        result_growing = model.compute_phenological_state(
            {
                'air_temperature_max': temp + 5.0,
                'air_temperature_min': temp - 5.0,
                "day_of_year": doy_growing,
                "air_temperature": temp,
            },
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert result_growing.phenology_phase in [PhenologyPhase.LEAF_EXPANSION, PhenologyPhase.FULL_LEAF]
        assert result_growing.growth_factor > 0.8
        
        # Test after senescence onset (day 300)
        doy_senescence = torch.tensor([300.0])
        temp_cold = torch.tensor([5.0])
        
        result_senescence = model.compute_phenological_state(
            {
                'air_temperature_max': temp_cold + 3.0,
                'air_temperature_min': temp_cold - 3.0,
                "day_of_year": doy_senescence,
                "air_temperature": temp_cold,
            },
            dt=get_physical_constant("seconds_per_day"),
        )

        assert result_senescence.phenology_phase in [PhenologyPhase.SENESCENCE, PhenologyPhase.ABSCISSION]
        assert result_senescence.growth_factor < result_growing.growth_factor

    def test_static_phenology_lai_calculation(self, static_phenology_config):
        """Test LAI calculation in static model"""
        model = StaticPhenology(config=static_phenology_config)
        
        # Test peak growing season
        doy_peak = torch.tensor([180.0])  # Mid-summer
        temp = torch.tensor([25.0])
        
        result = model.compute_phenological_state(
            {
                'air_temperature_max': temp + 5.0,
                'air_temperature_min': temp - 5.0,
                "day_of_year": doy_peak,
                "air_temperature": temp,
            },
            dt=get_physical_constant("seconds_per_day"),
        )
        
        # LAI should be close to maximum during peak season
        assert result.leaf_area_index < model.phenology_params.max_lai * 1.1  # Allow small tolerance
        assert result.leaf_area_index > model.phenology_params.max_lai * 0.8
        
    def test_static_phenology_seasonal_cycle(self, static_phenology_config, seasonal_environment_data):
        """Test complete seasonal cycle"""
        model = StaticPhenology(config=static_phenology_config)
        
        results = []
        for i in range(0, 365, 10):  # Sample every 10 days
            doy = torch.tensor([float(i + 1)])
            temp = seasonal_environment_data['air_temperature'][i:i+1]
            temp_max = seasonal_environment_data['air_temperature_max'][i:i+1]
            temp_min = seasonal_environment_data['air_temperature_min'][i:i+1]
            
            result = model.compute_phenological_state(
                {
                    'air_temperature_max': temp_max,
                    'air_temperature_min': temp_min,
                    "day_of_year": doy,
                    "air_temperature": temp,
                },
                dt=get_physical_constant("seconds_per_day"),
            )
            
            results.append({
                'doy': doy.item(),
                'phase': result.phenology_phase,
                'growth_index': result.growth_factor.item(),
                'lai': result.leaf_area_index.item()
            })
        
        # Check that we have proper seasonal progression
        phases = [r['phase'] for r in results]
        
        # Should start dormant
        assert phases[0] == PhenologyPhase.DORMANCY
        
        # Should have growing phase in summer
        summer_phases = phases[15:25]  # Around days 150-250
        assert any(p in [PhenologyPhase.LEAF_EXPANSION, PhenologyPhase.FULL_LEAF] for p in summer_phases)
        
        # Should end with senescence/dormancy
        assert phases[-1] in [PhenologyPhase.SENESCENCE, PhenologyPhase.ABSCISSION, PhenologyPhase.DORMANCY]


class TestGDDPhenology:
    """Test suite for GDDPhenology model"""
    
    def test_gdd_phenology_initialization(self, gdd_phenology_config):
        """Test GDD phenology model initialization"""
        model = GDDPhenology(config=gdd_phenology_config)
        
        assert model.metadata.name == 'gdd_phenology'
        assert hasattr(model.phenology_params, 'gdd_base_temp')
        assert hasattr(model.phenology_params, 'gdd_threshold_start')
        assert hasattr(model.phenology_params, 'gdd_threshold_end')
        assert model.phenology_params.gdd_base_temp == 5.0
        assert model.phenology_params.gdd_threshold_start == 200.0
        assert model.phenology_params.gdd_threshold_end == 1200.0
        assert "phenology_gdd_accumulated" in model.metadata.input_variables
        assert "phenology_gdd_previous_day_of_year" in model.metadata.input_variables
        assert "phenology_gdd_accumulated" in model.metadata.output_variables
        assert "phenology_gdd_previous_day_of_year" in model.metadata.output_variables
        assert not hasattr(model, "accumulated_gdd")
        assert not hasattr(model, "_last_gdd_day_of_year")
        init_source = inspect.getsource(GDDPhenology.__init__)
        assert "del self.accumulated_gdd" not in init_source
        assert "temporary compatibility surface" not in init_source
        source = inspect.getsource(GDDPhenology.compute_phenological_state)
        assert "temperature > 100" not in source
        assert "temp_max_raw > 100" not in source
        assert "temp_min_raw > 100" not in source
        
    def test_gdd_accumulation(self, gdd_phenology_config):
        """Test GDD accumulation logic"""
        model = GDDPhenology(config=gdd_phenology_config)
        
        # Test temperature series
        kelvin_offset = get_physical_constant("T_FREEZE")
        temp_max_celsius = torch.tensor([15.0, 18.0, 22.0, 20.0])
        temp_min_celsius = torch.tensor([8.0, 10.0, 14.0, 12.0])
        temp_max = temp_max_celsius + kelvin_offset
        temp_min = temp_min_celsius + kelvin_offset
        temp_avg = (temp_max + temp_min) / 2.0
        doy = torch.tensor([100.0, 101.0, 102.0, 103.0])
        accumulated_gdd = torch.tensor(0.0)
        previous_day = torch.tensor(99.0)
        
        results = []
        for i in range(len(doy)):
            result = model.compute_phenological_state(
                {
                    'air_temperature_max': temp_max[i:i+1],
                    'air_temperature_min': temp_min[i:i+1],
                    'phenology_gdd_accumulated': accumulated_gdd,
                    'phenology_gdd_previous_day_of_year': previous_day,
                    "day_of_year": doy[i:i+1],
                    "air_temperature": temp_avg[i:i+1],
                },
                dt=torch.tensor(
                        get_physical_constant("seconds_per_day")
                    ),
            )
            results.append(result)
            accumulated_gdd = result.additional_outputs[
                'phenology_gdd_accumulated'
            ]
            previous_day = result.additional_outputs[
                'phenology_gdd_previous_day_of_year'
            ]
        
        # Check GDD accumulation increases
        gdd_values = [
            r.additional_outputs['phenology_gdd_accumulated'].item()
            for r in results
        ]
        assert all(gdd_values[i] <= gdd_values[i+1] for i in range(len(gdd_values)-1))

        old_output_names = {
            'gdd_daily',
            'gdd_accumulated',
            'gdd_threshold_start',
            'gdd_threshold_end',
        }
        for result in results:
            assert old_output_names.isdisjoint(result.additional_outputs)
        
        # Check daily GDD calculation
        expected_daily_gdd = [
            max(
                0,
                (
                    temp_max_celsius[i].item()
                    + temp_min_celsius[i].item()
                ) / 2.0 - model.phenology_params.gdd_base_temp,
            )
            for i in range(len(temp_avg))
        ]
        
        daily_gdd_values = [
            r.diagnostic_variables['daily_gdd'].item() for r in results
        ]
        for i, (expected, actual) in enumerate(zip(expected_daily_gdd, daily_gdd_values)):
            assert abs(expected - actual) < 1e-6, f"Day {i}: expected {expected}, got {actual}"
            
    def test_gdd_phenological_transitions(self, gdd_phenology_config):
        """Test phenological transitions based on GDD thresholds"""
        model = GDDPhenology(config=gdd_phenology_config)
        
        # Simulate spring warming with gradual GDD accumulation
        base_temp = model.phenology_params.gdd_base_temp
        
        # Low GDD - should be dormant
        kelvin_offset = get_physical_constant("T_FREEZE")
        low_gdd_temp = torch.tensor([base_temp + 2.0 + kelvin_offset])
        doy_early = torch.tensor([90.0])
        
        result_early = model.compute_phenological_state(
            {
                'air_temperature_max': low_gdd_temp + 3.0,
                'air_temperature_min': low_gdd_temp - 3.0,
                'phenology_gdd_accumulated': torch.tensor(0.0),
                'phenology_gdd_previous_day_of_year': torch.tensor(89.0),
                "day_of_year": doy_early,
                "air_temperature": low_gdd_temp,
            },
            dt=torch.tensor(
                    get_physical_constant("seconds_per_day")
                ),
        )
        
        assert result_early.phenology_phase == PhenologyPhase.DORMANCY
        
        high_gdd = torch.tensor([800.0])  # Between start and end thresholds
        base_kelvin = torch.tensor([base_temp + kelvin_offset])
        result_growing = model.compute_phenological_state(
            {
                'air_temperature_max': base_kelvin,
                'air_temperature_min': base_kelvin,
                'phenology_gdd_accumulated': high_gdd,
                'phenology_gdd_previous_day_of_year': torch.tensor(149.0),
                "day_of_year": torch.tensor([150.0]),
                "air_temperature": base_kelvin,
            },
            dt=torch.tensor(86400.0),
        )

        assert result_growing.phenology_phase in [PhenologyPhase.LEAF_EXPANSION, PhenologyPhase.FULL_LEAF]
            
    def test_gdd_lai_response(self, gdd_phenology_config):
        """Test LAI response to GDD accumulation"""
        model = GDDPhenology(config=gdd_phenology_config)
        
        # Test different GDD levels and their LAI response
        test_gdds = [100.0, 400.0, 800.0, 1400.0]  # Below, at start, middle, above end
        base_kelvin = torch.tensor([
            model.phenology_params.gdd_base_temp
            + get_physical_constant("T_FREEZE")
        ])
        
        for gdd_val in test_gdds:
            result = model.compute_phenological_state(
                {
                    'air_temperature_max': base_kelvin,
                    'air_temperature_min': base_kelvin,
                    'phenology_gdd_accumulated': torch.tensor([gdd_val]),
                    'phenology_gdd_previous_day_of_year': torch.tensor(149.0),
                    "day_of_year": torch.tensor([150.0]),
                    "air_temperature": base_kelvin,
                },
                dt=torch.tensor(86400.0),
            )
                
            # LAI should increase with GDD up to threshold, then potentially decrease
            assert result.leaf_area_index >= 0.0
            assert result.leaf_area_index <= model.phenology_params.max_lai * 1.1
                
            if gdd_val < model.phenology_params.gdd_threshold_start:
                assert result.leaf_area_index < model.phenology_params.max_lai * 0.3
            elif gdd_val > model.phenology_params.gdd_threshold_end:
                assert result.phenology_phase == PhenologyPhase.ABSCISSION
                    
    def test_gdd_cold_temperature_handling(self, gdd_phenology_config):
        """Test handling of cold temperatures (negative GDD)"""
        model = GDDPhenology(config=gdd_phenology_config)
        
        # Test with temperatures below base temperature
        cold_temp = torch.tensor([
            model.phenology_params.gdd_base_temp
            - 5.0
            + get_physical_constant("T_FREEZE")
        ])
        doy = torch.tensor([100.0])
        
        result = model.compute_phenological_state(
            {
                'air_temperature_max': cold_temp + 2.0,
                'air_temperature_min': cold_temp - 2.0,
                'phenology_gdd_accumulated': torch.tensor(0.0),
                'phenology_gdd_previous_day_of_year': torch.tensor(99.0),
                "day_of_year": doy,
                "air_temperature": cold_temp,
            },
            dt=torch.tensor(86400.0),
        )
        
        # Daily GDD should be zero or negative (clamped to zero)
        assert 'gdd_daily' not in result.additional_outputs
        daily_gdd = result.diagnostic_variables['daily_gdd']
        assert daily_gdd >= 0.0, "Daily GDD should never be negative"
        assert daily_gdd < 1e-6, "Daily GDD should be effectively zero for cold temperatures"


class TestGSIPhenology:
    """Test suite for GSIPhenology model"""
    
    def test_gsi_phenology_initialization(self, gsi_phenology_config):
        """Test GSI phenology model initialization"""
        model = GSIPhenology(config=gsi_phenology_config)

        assert model.metadata.name == 'gsi_phenology'
        assert hasattr(model.phenology_params, 'gsi_tmin_threshold')
        assert hasattr(model.phenology_params, 'gsi_vpd_max')
        assert hasattr(model.phenology_params, 'gsi_onset_threshold')
        assert model.phenology_params.gsi_tmin_threshold == -2.0
        assert model.phenology_params.gsi_vpd_max == 4100.0
        assert model.phenology_params.gsi_onset_threshold == 0.5

        inputs = set(model.metadata.input_variables)
        outputs = set(model.metadata.output_variables)
        for key in GSI_TEMPORAL_STATE_KEYS:
            assert key in inputs
            assert key in outputs
            assert model.OUTPUT_UNITS[key] == "dimensionless"
        assert "air_temperature_min" in inputs
        assert "temperature_min" not in inputs
        assert "soil_water_potential" not in inputs
        assert "soil_water" not in model.metadata.coupling_interfaces
        assert "gsi_swp_wilting" not in model.metadata.parameters
        assert "gsi_swp_optimal" not in model.metadata.parameters

        for hidden_state in (
            "gsi_history",
            "gsi_history_idx",
            "gsi_history_count",
            "in_growing_season",
            "onset_triggered",
            "offset_triggered",
            "last_gsi_values",
        ):
            assert not hasattr(model, hidden_state)
        assert not hasattr(model, "reset_gsi_memory")
        init_source = inspect.getsource(GSIPhenology.__init__)
        assert "del self.last_gsi_values" not in init_source
        assert "temporary compatibility surface" not in init_source
        
    def test_gsi_component_calculations(self, gsi_phenology_config):
        """Use the canonical three-factor Jolly instantaneous GSI."""
        model = GSIPhenology(config=gsi_phenology_config)
        freeze = get_physical_constant("T_FREEZE")
        temp_min = torch.tensor([freeze + 3.0])
        vpd = torch.tensor([2000.0])  # Moderate
        daylen = torch.tensor([45000.0])  # Long day (12.5 hours)

        environment = {
            'air_temperature_min': temp_min,
            'vapor_pressure_deficit': vpd,
            'daylength': daylen,
            **_gsi_temporal_state(model),
        }
        result = model.compute_phenological_state(
            {**environment, "day_of_year": torch.tensor([150.0]), "air_temperature": torch.tensor([freeze + 18.0])},
            dt=torch.tensor(get_physical_constant("seconds_per_day")),
        )

        gsi = result.additional_outputs[
            'diagnostics.phenology.growing_season_index'
        ]
        assert gsi.item() == pytest.approx(0.46875)
        assert set(result.additional_outputs) == {
            *GSI_TEMPORAL_STATE_KEYS,
            'diagnostics.phenology.growing_season_index',
        }
        assert result.additional_outputs['phenology_gsi_history'][0].item() == (
            pytest.approx(0.46875)
        )
        assert result.additional_outputs['phenology_gsi_history_index'].item() == 1.0
        assert result.additional_outputs['phenology_gsi_history_count'].item() == 1.0
            
    def test_gsi_limiting_factors(self, gsi_phenology_config):
        """Tmin, VPD, and photoperiod are the only GSI limitations."""
        freeze = get_physical_constant("T_FREEZE")

        def evaluate(tmin_c: float, vpd: float, daylength: float) -> torch.Tensor:
            model = GSIPhenology(config=gsi_phenology_config)
            result = model.compute_phenological_state(
                {
                    'air_temperature_min': torch.tensor([freeze + tmin_c]),
                    'vapor_pressure_deficit': torch.tensor([vpd]),
                    'daylength': torch.tensor([daylength]),
                    **_gsi_temporal_state(model),
                    "day_of_year": torch.tensor([150.0]),
                    "air_temperature": torch.tensor([freeze + 18.0]),
                },
                dt=torch.tensor(
                        get_physical_constant("seconds_per_day")
                    ),
            )
            return result.additional_outputs[
                'diagnostics.phenology.growing_season_index'
            ]

        baseline_gsi = evaluate(5.0, 900.0, 50000.0)
        gsi_cold = evaluate(-5.0, 900.0, 50000.0)
        gsi_high_vpd = evaluate(5.0, 5000.0, 50000.0)
        gsi_short_day = evaluate(5.0, 900.0, 28000.0)

        assert baseline_gsi.item() == pytest.approx(1.0)
        assert gsi_cold < baseline_gsi
        assert gsi_high_vpd < baseline_gsi
        assert gsi_short_day < baseline_gsi
        
    def test_gsi_phenological_thresholds(self, gsi_phenology_config):
        """External hysteresis state controls onset and offset transitions."""
        model = GSIPhenology(config=gsi_phenology_config)
        freeze = get_physical_constant("T_FREEZE")
        favorable_conditions = {
            'air_temperature_min': torch.tensor([freeze + 5.0]),
            'vapor_pressure_deficit': torch.tensor([900.0]),
            'daylength': torch.tensor([50000.0]),
            **_gsi_temporal_state(model),
        }
        result_favorable = model.compute_phenological_state(
            {**favorable_conditions, "day_of_year": torch.tensor([130.0]), "air_temperature": torch.tensor([freeze + 20.0])},
            dt=torch.tensor(get_physical_constant("seconds_per_day")),
        )
        assert result_favorable.phenology_phase is PhenologyPhase.LEAF_EXPANSION
        assert result_favorable.growth_factor.item() == pytest.approx(1.0)
        assert result_favorable.additional_outputs[
            'phenology_gsi_in_growing_season'
        ].item() == 1.0

        full_low_history = torch.zeros(model.memory_days)
        limiting_conditions = {
            'air_temperature_min': torch.tensor([freeze - 5.0]),
            'vapor_pressure_deficit': torch.tensor([5000.0]),
            'daylength': torch.tensor([28000.0]),
            **_gsi_temporal_state(
                model,
                history=full_low_history,
                index=0,
                count=model.memory_days,
                in_growing_season=True,
            ),
        }
        result_limiting = model.compute_phenological_state(
            {**limiting_conditions, "day_of_year": torch.tensor([250.0]), "air_temperature": torch.tensor([freeze + 10.0])},
            dt=torch.tensor(get_physical_constant("seconds_per_day")),
        )
        assert result_limiting.phenology_phase is PhenologyPhase.SENESCENCE
        assert result_limiting.additional_outputs[
            'phenology_gsi_in_growing_season'
        ].item() == 0.0
        
    def test_gsi_seasonal_memory_effect(self, gsi_phenology_config):
        """Identical forcing responds only to StateManager-owned history."""
        freeze = get_physical_constant("T_FREEZE")
        low_model = GSIPhenology(config=gsi_phenology_config)
        high_model = GSIPhenology(config=gsi_phenology_config)
        low_history = torch.zeros(low_model.memory_days)
        high_history = torch.ones(high_model.memory_days)
        low_history_before = low_history.clone()
        high_history_before = high_history.clone()
        forcing = {
            'air_temperature_min': torch.tensor([freeze + 5.0]),
            'vapor_pressure_deficit': torch.tensor([900.0]),
            'daylength': torch.tensor([50000.0]),
        }

        low_result = low_model.compute_phenological_state(
            {
                **forcing,
                **_gsi_temporal_state(
                    low_model,
                    history=low_history,
                    count=low_model.memory_days,
                ),
                "day_of_year": torch.tensor([140.0]),
                "air_temperature": torch.tensor([freeze + 18.0]),
            },
            dt=get_physical_constant("seconds_per_day"),
        )
        high_result = high_model.compute_phenological_state(
            {
                **forcing,
                **_gsi_temporal_state(
                    high_model,
                    history=high_history,
                    count=high_model.memory_days,
                    in_growing_season=True,
                ),
                "day_of_year": torch.tensor([140.0]),
                "air_temperature": torch.tensor([freeze + 18.0]),
            },
            dt=get_physical_constant("seconds_per_day"),
        )

        torch.testing.assert_close(low_history, low_history_before)
        torch.testing.assert_close(high_history, high_history_before)
        assert low_result.phenology_phase is PhenologyPhase.DORMANCY
        assert high_result.phenology_phase is PhenologyPhase.FULL_LEAF
        assert low_result.growing_season_index.item() == pytest.approx(
            1.0 / low_model.memory_days
        )
        assert high_result.growing_season_index.item() == pytest.approx(1.0)

    def test_gsi_forward_emits_full_state_updates_at_daily_cadence(
        self,
        gsi_phenology_config,
        standard_vegetation_state,
    ):
        """GSI temporal state is full-state REPLACE at one daily sample."""
        model = GSIPhenology(config=gsi_phenology_config)
        freeze = get_physical_constant("T_FREEZE")
        seconds_per_day = get_physical_constant("seconds_per_day")
        state = {
            **standard_vegetation_state,
            'day_of_year': torch.tensor(140.0),
            'air_temperature': torch.tensor(freeze + 18.0),
            'air_temperature_min': torch.tensor(freeze + 5.0),
            'vapor_pressure_deficit': torch.tensor(900.0),
            'daylength': torch.tensor(50000.0),
            **_gsi_temporal_state(model),
        }

        updates = model(state, time=0.0, dt=seconds_per_day)
        for key in GSI_TEMPORAL_STATE_KEYS:
            update = updates[key]
            assert update.mode is UpdateMode.REPLACE
            assert update.per_timestep is True
            assert update.unit == "dimensionless"
            assert update.source_functor == "gsi_phenology"
        assert 'diagnostics.phenology.growing_season_index' in updates
        for alias in ('gsi', 'gsi_memory', 'f_temp', 'f_vpd', 'f_daylen', 'f_swp'):
            assert alias not in updates

        with pytest.raises(ExecutionError, match="daily"):
            model(state, time=0.0, dt=3600.0)


class TestCropPhenology:
    """Test suite for CropPhenology model"""

    def test_crop_phenology_initialization(self, crop_phenology_config):
        """Crop temporal memory belongs to StateManager, not the Functor."""
        model = CropPhenology(
            config=crop_phenology_config['config'],
            phenology_parameters=crop_phenology_config['params'],
        )

        assert model.metadata.name == 'crop_phenology'
        assert hasattr(model.phenology_params, 'crop_phu_total')
        assert hasattr(model.phenology_params, 'crop_hi_max')
        assert hasattr(model.phenology_params, 'crop_sowing_doy')
        assert hasattr(model.phenology_params, 'crop_harvest_doy')
        assert model.phenology_params.crop_phu_total == 500.0
        assert model.phenology_params.crop_hi_max == 0.45
        assert model.phenology_params.crop_sowing_doy == 120
        assert model.phenology_params.crop_harvest_doy == 280

        inputs = set(model.metadata.input_variables)
        outputs = set(model.metadata.output_variables)
        for key, unit in CROP_TEMPORAL_STATE_UNITS.items():
            assert key in inputs
            assert key in outputs
            assert model.OUTPUT_UNITS[key] == unit

        for hidden_state in (
            'accumulated_phu',
            'development_index',
            'harvest_index',
            'is_sown',
            'is_harvested',
            'vernalization_days',
        ):
            assert not hasattr(model, hidden_state)
        for hidden_mutator in (
            'sow',
            '_sow_crop',
            '_harvest_crop',
            'reset_crop_cycle',
            '_get_accumulated_phu',
        ):
            assert not hasattr(model, hidden_mutator)

        source = inspect.getsource(CropPhenology.compute_phenological_state)
        assert 'temperature > 100' not in source
        assert 'min=0.1' not in source
        assert 'vernalization_min_factor' in source

        vernalization_floor = get_global_manager().get_parameter(
            'biosphere.phenology.crop.vernalization_min_factor'
        )
        assert vernalization_floor is not None
        assert vernalization_floor.units == 'dimensionless'
        assert vernalization_floor.dimension == 'dimensionless'

    def test_crop_phu_accumulation(self, crop_phenology_config):
        """Canonical Kelvin forcing advances external PHU with exact dt scaling."""
        model = CropPhenology(
            config=crop_phenology_config['config'],
            phenology_parameters=crop_phenology_config['params'],
        )

        state = _crop_temporal_state()
        temperatures_c = (15.0, 18.0, 22.0, 25.0, 20.0)
        expected_phu = (7.0, 17.0, 31.0, 48.0, 60.0)
        for offset, (temperature_c, expected) in enumerate(
            zip(temperatures_c, expected_phu)
        ):
            input_state = state
            state_before = {
                key: value.clone() for key, value in input_state.items()
            }
            result, state = _advance_crop(
                model,
                input_state,
                day_of_year=120.0 + offset,
                temperature_c=temperature_c,
            )
            assert state['phenology_crop_accumulated_phu'].item() == pytest.approx(
                expected
            )
            assert result.additional_outputs[
                'diagnostics.phenology.accumulated_phu'
            ].item() == pytest.approx(expected)
            assert result.diagnostic_variables['development_index'].item() == (
                pytest.approx(expected / model.phenology_params.crop_phu_total)
            )
            assert state['phenology_crop_is_sown'].item() == 1.0
            assert state['phenology_crop_is_harvested'].item() == 0.0
            assert state['phenology_crop_vernalization_days'].item() == 0.0
            for key, value in state_before.items():
                torch.testing.assert_close(input_state[key], value)
            for alias in ('phu_accumulated', 'phu_fraction', 'phu_total', 'is_sown'):
                assert alias not in result.additional_outputs

        seconds_per_day = float(get_physical_constant('seconds_per_day'))
        full_result, _ = _advance_crop(
            CropPhenology(
                config=crop_phenology_config['config'],
                phenology_parameters=crop_phenology_config['params'],
            ),
            _crop_temporal_state(),
            day_of_year=120.0,
            temperature_c=20.0,
            dt=seconds_per_day,
        )
        half_model = CropPhenology(
            config=crop_phenology_config['config'],
            phenology_parameters=crop_phenology_config['params'],
        )
        half_result, half_state = _advance_crop(
            half_model,
            _crop_temporal_state(),
            day_of_year=120.0,
            temperature_c=20.0,
            dt=seconds_per_day / 2.0,
        )
        assert half_result.additional_outputs[
            'phenology_crop_accumulated_phu'
        ].item() == pytest.approx(6.0)
        second_half_result, _ = _advance_crop(
            half_model,
            half_state,
            day_of_year=120.0,
            temperature_c=20.0,
            dt=seconds_per_day / 2.0,
        )
        assert second_half_result.additional_outputs[
            'phenology_crop_accumulated_phu'
        ].item() == pytest.approx(
            full_result.additional_outputs[
                'phenology_crop_accumulated_phu'
            ].item()
        )

    def test_crop_forward_returns_canonical_temporal_updates(
        self,
        crop_phenology_config,
        standard_vegetation_state,
    ):
        """Crop temporal outputs are explicit full-state UpdateItems."""
        model = CropPhenology(
            config=crop_phenology_config['config'],
            phenology_parameters=crop_phenology_config['params'],
        )
        seconds_per_day = float(get_physical_constant('seconds_per_day'))
        temporal_state = _crop_temporal_state()
        temporal_before = {
            key: value.clone() for key, value in temporal_state.items()
        }
        state = {
            **standard_vegetation_state,
            'day_of_year': torch.tensor(120.0),
            'air_temperature': torch.tensor(
                get_physical_constant('T_FREEZE') + 20.0
            ),
            **temporal_state,
        }

        updates = model(state, time=0.0, dt=seconds_per_day)

        expected_values = {
            'phenology_crop_accumulated_phu': 12.0,
            'phenology_crop_is_sown': 1.0,
            'phenology_crop_is_harvested': 0.0,
            'phenology_crop_vernalization_days': 0.0,
        }
        for key, expected in expected_values.items():
            update = updates[key]
            assert update.mode is UpdateMode.REPLACE
            assert update.per_timestep is True
            assert update.unit == CROP_TEMPORAL_STATE_UNITS[key]
            assert update.source_functor == 'crop_phenology'
            assert update.value.item() == pytest.approx(expected)
            torch.testing.assert_close(state[key], temporal_before[key])
        for alias in ('phu_accumulated', 'phu_fraction', 'phu_total', 'is_sown'):
            assert alias not in updates

    def test_crop_harvest_index_development(self, crop_phenology_config):
        """Harvest index is derived from external PHU without a hidden override."""
        cases = (
            (0.2, 0.0),
            (0.4, 0.0),
            (0.6, 0.15),
            (0.8, 0.30),
            (1.0, 0.45),
        )
        for development_index, expected_hi in cases:
            target_phu = development_index * 500.0
            result, state = _advance_crop(
                CropPhenology(
                    config=crop_phenology_config['config'],
                    phenology_parameters=crop_phenology_config['params'],
                ),
                _crop_temporal_state(
                    accumulated_phu=target_phu - 12.0,
                    is_sown=True,
                ),
                day_of_year=180.0,
                temperature_c=20.0,
            )
            assert state['phenology_crop_accumulated_phu'].item() == pytest.approx(
                target_phu
            )
            assert result.additional_outputs['harvest_index'].item() == (
                pytest.approx(expected_hi, abs=1.0e-6)
            )

    def test_crop_phenological_stages(self, crop_phenology_config):
        """Sowing, harvest, and annual reset preserve canonical cycle state."""
        model = CropPhenology(
            config=crop_phenology_config['config'],
            phenology_parameters=crop_phenology_config['params'],
        )

        result_before, state_before = _advance_crop(
            model,
            _crop_temporal_state(),
            day_of_year=100.0,
            temperature_c=15.0,
        )
        assert result_before.phenology_phase == PhenologyPhase.DORMANCY
        assert all(value.item() == 0.0 for value in state_before.values())

        sowing_result, sowing_state = _advance_crop(
            model,
            state_before,
            day_of_year=120.0,
            temperature_c=20.0,
        )
        assert sowing_result.phenology_phase is PhenologyPhase.BUD_BREAK
        assert sowing_state['phenology_crop_accumulated_phu'].item() == (
            pytest.approx(12.0)
        )
        assert sowing_state['phenology_crop_is_sown'].item() == 1.0
        assert sowing_state['phenology_crop_is_harvested'].item() == 0.0

        harvest_result, harvest_state = _advance_crop(
            model,
            _crop_temporal_state(accumulated_phu=462.0, is_sown=True),
            day_of_year=280.0,
            temperature_c=20.0,
        )
        assert harvest_result.phenology_phase is PhenologyPhase.DORMANCY
        assert harvest_state['phenology_crop_accumulated_phu'].item() == (
            pytest.approx(462.0)
        )
        assert harvest_result.additional_outputs[
            'diagnostics.phenology.accumulated_phu'
        ].item() == pytest.approx(462.0)
        assert harvest_state['phenology_crop_is_sown'].item() == 1.0
        assert harvest_state['phenology_crop_is_harvested'].item() == 1.0

        reset_result, reset_state = _advance_crop(
            model,
            harvest_state,
            day_of_year=1.0,
            temperature_c=20.0,
        )
        assert reset_result.phenology_phase is PhenologyPhase.DORMANCY
        assert all(value.item() == 0.0 for value in reset_state.values())

    def test_crop_lai_development(self, crop_phenology_config):
        """LAI follows the development index derived from external PHU."""
        cases = (
            (0.1, PhenologyPhase.BUD_BREAK, 0.1),
            (0.3, PhenologyPhase.LEAF_EXPANSION, 5.0),
            (0.5, PhenologyPhase.FLOWERING, 3.5714288),
            (0.7, PhenologyPhase.SENESCENCE, 2.1428573),
            (0.9, PhenologyPhase.SENESCENCE, 0.7142859),
            (1.0, PhenologyPhase.ABSCISSION, 0.0),
        )
        for development_index, expected_phase, expected_lai in cases:
            target_phu = development_index * 500.0
            result, _ = _advance_crop(
                CropPhenology(
                    config=crop_phenology_config['config'],
                    phenology_parameters=crop_phenology_config['params'],
                ),
                _crop_temporal_state(
                    accumulated_phu=target_phu - 12.0,
                    is_sown=True,
                ),
                day_of_year=180.0,
                temperature_c=20.0,
            )
            assert result.phenology_phase is expected_phase
            assert result.leaf_area_index.item() == pytest.approx(
                expected_lai,
                abs=1.0e-6,
            )

    def test_cross_year_winter_wheat_cycle(
        self,
        winter_wheat_phenology_config,
    ):
        """A cross-year crop is not harvested on its autumn sowing date."""
        model = CropPhenology(
            config=winter_wheat_phenology_config['config'],
            phenology_parameters=winter_wheat_phenology_config['params'],
        )
        state = _crop_temporal_state()
        sequence = (
            (270.0, 0.3, 1.0, 1.0, 0.0),
            (365.0, 0.6, 2.0, 1.0, 0.0),
            (1.0, 0.9, 3.0, 1.0, 0.0),
            (149.0, 1.2, 4.0, 1.0, 0.0),
            (150.0, 1.2, 4.0, 1.0, 1.0),
            (270.0, 0.3, 1.0, 1.0, 0.0),
        )
        for doy, phu, vern, sown, harvested in sequence:
            result, state = _advance_crop(
                model,
                state,
                day_of_year=doy,
                temperature_c=8.0,
            )
            assert state['phenology_crop_accumulated_phu'].item() == pytest.approx(
                phu,
                abs=1.0e-6,
            )
            assert state['phenology_crop_vernalization_days'].item() == (
                pytest.approx(vern)
            )
            assert state['phenology_crop_is_sown'].item() == sown
            assert state['phenology_crop_is_harvested'].item() == harvested
            assert result.additional_outputs[
                'diagnostics.phenology.accumulated_phu'
            ].item() == pytest.approx(phu, abs=1.0e-6)

    def test_crop_temporal_state_validation(self, crop_phenology_config):
        """Invalid external Crop temporal states fail before calculation."""
        model = CropPhenology(
            config=crop_phenology_config['config'],
            phenology_parameters=crop_phenology_config['params'],
        )
        invalid_cases = (
            (
                _crop_temporal_state(accumulated_phu=-1.0, is_sown=True),
                'phenology_crop_accumulated_phu',
            ),
            (
                _crop_temporal_state(is_sown=False, is_harvested=True),
                'cycle flags',
            ),
            (
                {
                    **_crop_temporal_state(is_sown=True),
                    'phenology_crop_is_sown': torch.tensor(0.5),
                },
                'phenology_crop_is_sown',
            ),
            (
                {
                    **_crop_temporal_state(is_sown=True),
                    'phenology_crop_vernalization_days': torch.tensor(float('nan')),
                },
                'phenology_crop_vernalization_days',
            ),
        )
        for state, message in invalid_cases:
            with pytest.raises(ValueError, match=message):
                _advance_crop(
                    model,
                    state,
                    day_of_year=140.0,
                    temperature_c=20.0,
                )


# Integration tests for all models
class TestPhenologyModelIntegration:
    """Integration tests across all phenology models"""
    
    def test_all_models_basic_functionality(self):
        """Test that all models can be instantiated and run basic operations"""
        configs = [
            ('static', {
                'phenology_parameters': {
                    'phenology_type': 'static',
                    'leaf_onset_doy': 120, 'leaf_offset_doy': 280, 'latitude': 45.0
                }
            }),
            ('gdd', {
                'phenology_parameters': {
                    'phenology_type': 'gdd',
                    'gdd_base_temp': -5.0, 'gdd_threshold_start': 200.0,
                    'gdd_threshold_end': 1200.0, 'latitude': 45.0
                }
            }),
            ('gsi', {
                'phenology_parameters': {
                    'phenology_type': 'gsi',
                    'gsi_tmin_threshold': -2.0, 'gsi_onset_threshold': 0.5,
                    'latitude': 45.0
                }
            }),
            ('crop', {
                'phenology_parameters': {
                    'phenology_type': 'crop',
                    'crop_phu_total': 500.0, 'crop_sowing_doy': 120,
                    'crop_harvest_doy': 280, 'latitude': 40.0
                }
            })
        ]
        
        models = {
            'static': StaticPhenology,
            'gdd': GDDPhenology,
            'gsi': GSIPhenology,
            'crop': CropPhenology
        }
        
        for model_type, config in configs:
            model = models[model_type](config=config)
            
            # Test basic functionality
            assert model.metadata.name.endswith('_phenology')
            assert hasattr(model, 'compute_phenological_state')
            assert hasattr(model, 'forward')
            assert hasattr(model, 'forward')
            
            # Test with model-specific canonical inputs.
            temperature = torch.tensor([20.0])
            environment = {
                'air_temperature_max': torch.tensor([25.0]),
                'air_temperature_min': torch.tensor([15.0]),
            }
            if model_type == 'gdd':
                kelvin_offset = get_physical_constant("T_FREEZE")
                temperature = torch.tensor([20.0 + kelvin_offset])
                environment.update({
                    'air_temperature_max': torch.tensor([25.0 + kelvin_offset]),
                    'air_temperature_min': torch.tensor([15.0 + kelvin_offset]),
                    'phenology_gdd_accumulated': torch.tensor(0.0),
                    'phenology_gdd_previous_day_of_year': torch.tensor(179.0),
                })
            elif model_type == 'gsi':
                kelvin_offset = get_physical_constant("T_FREEZE")
                temperature = torch.tensor([20.0 + kelvin_offset])
                environment = {
                    'air_temperature_min': torch.tensor(
                        [15.0 + kelvin_offset]
                    ),
                    'vapor_pressure_deficit': torch.tensor([2000.0]),
                    'daylength': torch.tensor([45000.0]),
                    **_gsi_temporal_state(model),
                }
            elif model_type == 'crop':
                kelvin_offset = get_physical_constant("T_FREEZE")
                temperature = torch.tensor([20.0 + kelvin_offset])
                environment = {
                    **_crop_temporal_state(is_sown=True),
                }

            result = model.compute_phenological_state(
                {**environment, "day_of_year": torch.tensor([180.0]), "air_temperature": temperature},
                dt=get_physical_constant("seconds_per_day"),
            )
            
            # All models should return valid PhenologyOutput
            assert isinstance(result, PhenologyOutput)
            assert hasattr(result, 'phenology_phase')
            assert hasattr(result, 'growth_factor')
            assert hasattr(result, 'leaf_area_index')
            assert hasattr(result, 'senescence_rate')
            
            # Check output ranges
            assert result.growth_factor >= 0.0 and result.growth_factor <= 1.0
            assert result.leaf_area_index >= 0.0
            assert result.senescence_rate >= 0.0


class _RecordingInputs(dict):
    """Input mapping that records every key a phenology model reads or probes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.accessed.add(key)
        return super().__contains__(key)

    def get(self, key, *args):
        self.accessed.add(key)
        return super().get(key, *args)


def _gdd_inputs(accumulated=150.0):
    freeze = get_physical_constant("T_FREEZE")
    return {
        "day_of_year": torch.tensor(150.0),
        "air_temperature": torch.tensor(freeze + 20.0),
        "air_temperature_max": torch.tensor(freeze + 26.0),
        "air_temperature_min": torch.tensor(freeze + 11.0),
        "phenology_gdd_accumulated": torch.tensor(accumulated),
        "phenology_gdd_previous_day_of_year": torch.tensor(149.0),
    }


class TestDeclaredStateInputsOnly:
    """Phenology reads its scientific inputs only from the metadata-declared state inputs;
    the step length is the simulator dt keyword. No environment mapping, time_step entry
    or simulation_time control is passed."""

    def test_signatures_take_state_and_keyword_dt(self):
        from relic.scientific.biosphere.phenology.base import PhenologyBase

        for owner in (PhenologyBase, StaticPhenology, GDDPhenology, GSIPhenology, CropPhenology):
            params = inspect.signature(owner.compute_phenological_state).parameters
            assert list(params) == ["self", "state", "dt"], owner.__name__
            assert params["dt"].kind is inspect.Parameter.KEYWORD_ONLY, owner.__name__
        core = inspect.signature(PhenologyBase._compute_rate_core).parameters
        assert list(core) == ["self", "state", "dt"]

    def test_models_read_only_declared_inputs(
        self,
        static_phenology_config,
        gdd_phenology_config,
        gsi_phenology_config,
        crop_phenology_config,
    ):
        freeze = get_physical_constant("T_FREEZE")
        day = float(get_physical_constant("seconds_per_day"))
        gsi = GSIPhenology(config=gsi_phenology_config)
        crop = CropPhenology(
            config=crop_phenology_config['config'],
            phenology_parameters=crop_phenology_config['params'],
        )
        cases = (
            (
                StaticPhenology(config=static_phenology_config),
                {"day_of_year": torch.tensor([150.0]), "air_temperature": torch.tensor([freeze + 15.0])},
            ),
            (GDDPhenology(config=gdd_phenology_config), _gdd_inputs()),
            (
                gsi,
                {
                    "day_of_year": torch.tensor(140.0),
                    "air_temperature": torch.tensor(freeze + 18.0),
                    "air_temperature_min": torch.tensor(freeze + 5.0),
                    "vapor_pressure_deficit": torch.tensor(900.0),
                    "daylength": torch.tensor(50000.0),
                    **_gsi_temporal_state(gsi),
                },
            ),
            (
                crop,
                {
                    "day_of_year": torch.tensor([120.0]),
                    "air_temperature": torch.tensor([freeze + 20.0]),
                    **_crop_temporal_state(),
                },
            ),
        )
        for model, inputs in cases:
            recorded = _RecordingInputs(inputs)
            model.compute_phenological_state(recorded, dt=day)
            declared = set(model.metadata.input_variables)
            # Each law reads its calendar or temperature driver (GSI reads no day_of_year).
            assert recorded.accessed & {"day_of_year", "air_temperature"}, model.name
            assert recorded.accessed <= declared, (model.name, sorted(recorded.accessed - declared))

    def test_gdd_step_length_is_the_dt_keyword(self, gdd_phenology_config):
        model = GDDPhenology(config=gdd_phenology_config)
        day = float(get_physical_constant("seconds_per_day"))
        key = "phenology_gdd_accumulated"
        full = model.compute_phenological_state(_gdd_inputs(), dt=day)
        half = model.compute_phenological_state(_gdd_inputs(), dt=day / 2.0)
        increment = full.additional_outputs[key] - 150.0
        assert float(increment) > 0.0
        torch.testing.assert_close(half.additional_outputs[key] - 150.0, increment / 2.0)
        # An undeclared time_step entry is not an input; only the dt keyword counts.
        stale = model.compute_phenological_state(
            {**_gdd_inputs(), "time_step": torch.tensor(day / 2.0)}, dt=day
        )
        torch.testing.assert_close(stale.additional_outputs[key], full.additional_outputs[key])
        with pytest.raises(TypeError):
            model.compute_phenological_state(_gdd_inputs())

    def test_static_phenology_is_calendar_only(self, static_phenology_config):
        """StaticPhenology reads day_of_year alone: the temperature extrema the former
        environment mapping carried were never read."""
        model = StaticPhenology(config=static_phenology_config)
        freeze = get_physical_constant("T_FREEZE")
        day = float(get_physical_constant("seconds_per_day"))

        def run(doy, celsius):
            return model.compute_phenological_state(
                {"day_of_year": torch.tensor([doy]), "air_temperature": torch.tensor([freeze + celsius])},
                dt=day,
            )

        cold, warm = run(140.0, -10.0), run(140.0, 30.0)
        assert cold.phenology_phase == warm.phenology_phase == PhenologyPhase.LEAF_EXPANSION
        torch.testing.assert_close(cold.growth_factor, warm.growth_factor)
        torch.testing.assert_close(cold.leaf_area_index, warm.leaf_area_index)
        assert run(300.0, 30.0).phenology_phase != cold.phenology_phase


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

