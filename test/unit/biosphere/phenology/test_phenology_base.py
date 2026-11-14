"""
Unit Tests for Phenology Base Classes and Core Functionality

This test suite validates the core functionality of the phenology module,
including the abstract base class, parameter system, and utility functions.
All tests are designed to ensure mathematical accuracy, physical constraints,
and integration consistency with the RELIC framework.

Test Coverage:
    - PhenologyBase abstract class interface
    - Parameter validation and type safety
    - Utility function mathematical correctness
    - Physical constraint enforcement
    - Conservation law compliance
    - PyTorch tensor operations

References:
    - Jolly, W.M., et al. (2005). Global Change Biology, 11(4), 619-632
    - White, M.A., et al. (1997). Global Biogeochemical Cycles, 11(2), 217-234
    - McMaster, G.S. & Wilhelm, W.W. (1997). Agricultural and Forest Meteorology, 87(4), 291-300
"""

import inspect

import pytest
from pydantic import ValidationError
import torch
from typing import Dict, Any
import logging

# Import phenology module components
from relic.scientific.biosphere.phenology import (
    PhenologyBase, PhenologyParameters, PhenologyParameterSets,
    PhenologyPhase, PhenologyOutput
)
from relic.scientific.biosphere.phenology.utils import (
    compute_gdd, compute_gsi, compute_gsi_components, compute_daylength,
    thermal_time_accumulation, compute_chilling_hours,
    rolling_average, julian_day_to_date, date_to_julian_day, validate_environmental_data
)
from relic.core.state import UpdateItem, UpdateMode

# Set up logging for tests
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _parameters(**overrides):
    """Build a complete explicit payload for focused tests: the preset of the requested
    model (gsi unless phenology_type is given), which holds exactly that model's fields,
    with the overrides applied."""
    presets = {
        "gsi": PhenologyParameterSets.deciduous_tree_gsi,
        "static": PhenologyParameterSets.evergreen_tree_static,
        "gdd": PhenologyParameterSets.grassland_gdd,
        "crop": PhenologyParameterSets.maize_crop,
    }
    kind = overrides.get("phenology_type", "gsi")
    values = {
        name: value
        for name, value in presets[kind]().model_dump().items()
        if value is not None
    }
    values.update(overrides)
    return PhenologyParameters(**values)

# Test data and fixtures
@pytest.fixture
def basic_phenology_config():
    """Basic phenology configuration for testing"""
    return {
        'name': 'test_phenology',
        'phenology_type': 'gdd',
        'gdd_base_temp': 5.0,
        'gdd_threshold_start': 200.0,
        'gdd_threshold_end': 1200.0,
        'latitude': 45.0,
        'max_lai': 6.0,
        'leaf_turnover_rate': 0.25,
        'froot_turnover_rate': 1.0
    }

@pytest.fixture
def environmental_data():
    """Standard environmental data for testing"""
    batch_size = 10
    return {
        'air_temperature': torch.tensor([15.0, 18.0, 22.0, 20.0, 16.0, 12.0, 8.0, 5.0, 10.0, 14.0]),
        'air_temperature_max': torch.tensor([20.0, 23.0, 27.0, 25.0, 21.0, 17.0, 13.0, 10.0, 15.0, 19.0]),
        'air_temperature_min': torch.tensor([10.0, 13.0, 17.0, 15.0, 11.0, 7.0, 3.0, 0.0, 5.0, 9.0]),
        'vapor_pressure_deficit': torch.tensor([1500.0, 2000.0, 2500.0, 2200.0, 1800.0, 1200.0, 800.0, 500.0, 1000.0, 1400.0]),
        'daylength': torch.tensor([43200.0, 45000.0, 47000.0, 48000.0, 47500.0, 45500.0, 42000.0, 38000.0, 40000.0, 42500.0]),
        'soil_water_potential': torch.tensor([-0.05, -0.08, -0.12, -0.10, -0.06, -0.04, -0.02, -0.01, -0.03, -0.07]),
        'day_of_year': torch.tensor([120, 135, 150, 165, 180, 195, 210, 225, 240, 255])
    }

@pytest.fixture  
def vegetation_state():
    """Standard vegetation state for testing"""
    return {
        'leaf_carbon': torch.tensor([0.5, 0.8, 1.2, 1.5, 1.3, 1.0, 0.7, 0.4, 0.6, 0.9]),
        'froot_carbon': torch.tensor([0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.2, 0.3, 0.5]),
        'stem_carbon': torch.tensor([2.0, 2.5, 3.0, 3.2, 3.1, 2.8, 2.6, 2.3, 2.4, 2.7]),
        'leaf_nitrogen': torch.tensor([0.05, 0.08, 0.12, 0.15, 0.13, 0.10, 0.07, 0.04, 0.06, 0.09]),
        'froot_nitrogen': torch.tensor([0.03, 0.05, 0.07, 0.09, 0.08, 0.06, 0.04, 0.02, 0.03, 0.05]),
        'lai': torch.tensor([2.0, 3.5, 5.0, 6.0, 5.5, 4.0, 2.8, 1.5, 2.5, 3.8])
    }


class TestPhenologyParameters:
    """Test suite for phenology parameter system"""
    
    def test_phenology_parameters_creation(self):
        """Test basic parameter creation and validation"""
        params = _parameters(
            phenology_type='gdd',
            gdd_base_temp=5.0,
            gdd_threshold_start=200.0,
            gdd_threshold_end=1200.0,
        )

        assert params.phenology_type == 'gdd'
        assert params.gdd_base_temp == 5.0
        assert params.gdd_threshold_start == 200.0
        assert params.gdd_threshold_end == 1200.0

        # Fields not overridden come from the GDD preset
        preset = PhenologyParameterSets.grassland_gdd()
        assert params.max_lai == preset.max_lai
        assert params.leaf_turnover_rate == preset.leaf_turnover_rate
        assert params.froot_turnover_rate == preset.froot_turnover_rate

    def test_parameter_validation(self):
        """Test parameter validation and constraints"""
        # Test valid parameters
        valid_params = _parameters(
            phenology_type='gdd',
            gdd_base_temp=0.0,
            gdd_threshold_start=100.0,
            gdd_threshold_end=1500.0,
        )
        assert valid_params.gdd_base_temp >= -10.0

        # Test base temperature below its bound
        with pytest.raises(ValidationError):
            _parameters(phenology_type='gdd', gdd_base_temp=-15.0)

        # Test negative thresholds
        with pytest.raises(ValidationError):
            _parameters(phenology_type='gdd', gdd_threshold_start=-100.0)

        # Test threshold ordering
        with pytest.raises(ValidationError):
            _parameters(
                phenology_type='gdd',
                gdd_threshold_start=1500.0,
                gdd_threshold_end=1000.0,
            )

    def test_parameter_sets_creation(self):
        """Test predefined parameter sets"""
        # Test deciduous tree parameter set
        forest_params = PhenologyParameterSets.deciduous_tree_gsi()
        assert isinstance(forest_params, PhenologyParameters)
        assert forest_params.phenology_type == 'gsi'
        assert forest_params.max_lai >= 4.0
        
        # Test grassland parameter set
        grass_params = PhenologyParameterSets.grassland_gdd()
        assert isinstance(grass_params, PhenologyParameters)
        assert grass_params.phenology_type == 'gdd'
        assert grass_params.leaf_turnover_rate == pytest.approx(7.927e-09, rel=0, abs=1e-12)
        
        # Test crop parameter set
        crop_params = PhenologyParameterSets.maize_crop()
        assert isinstance(crop_params, PhenologyParameters)
        assert crop_params.phenology_type == 'crop'
        assert hasattr(crop_params, 'crop_phu_total')


class TestUtilityFunctions:
    """Test suite for phenology utility functions"""
    
    def test_compute_gdd(self):
        """Test growing degree day calculation"""
        # Test basic GDD calculation
        t_max = torch.tensor([20.0, 25.0, 30.0, 35.0], dtype=torch.float32)
        t_min = torch.tensor([10.0, 15.0, 20.0, 25.0], dtype=torch.float32)
        base_temp = 10.0

        gdd = compute_gdd(t_max, t_min, base_temp)
        expected = torch.tensor(
            [5.0, 10.0, 15.0, 20.0], dtype=gdd.dtype
        )  # (max+min)/2 - base for each day

        assert torch.allclose(gdd, expected, atol=1e-6)
        
        # Test with temperatures below base
        t_cold_max = torch.tensor([8.0, 5.0, 12.0], dtype=torch.float32)
        t_cold_min = torch.tensor([2.0, -9.0, 4.0], dtype=torch.float32)
        gdd_cold = compute_gdd(t_cold_max, t_cold_min, base_temp)
        expected_cold = torch.zeros_like(gdd_cold)

        assert torch.allclose(gdd_cold, expected_cold, atol=1e-6)
        
        # Test tensor properties
        assert gdd.dtype == torch.float32
        assert gdd.requires_grad == t_max.requires_grad
        
    def test_compute_daylength(self):
        """Test daylength calculation"""
        # Test at known latitude and day
        latitude = torch.tensor([45.0], dtype=torch.float32)  # 45掳N
        doy = torch.tensor([172], dtype=torch.float32)  # June 21 (summer solstice)

        daylength = compute_daylength(
            doy,
            latitude.item(),
            calendar_type="noleap",
        )

        # Test winter solstice
        doy_winter = torch.tensor([355], dtype=torch.float32)  # December 21
        daylength_winter = compute_daylength(
            doy_winter,
            latitude.item(),
            calendar_type="noleap",
        )

        # Winter daylength should be shorter than summer
        assert daylength_winter < daylength

        # Test equator
        latitude_eq = torch.tensor([0.0], dtype=torch.float32)
        daylength_eq = compute_daylength(
            doy,
            latitude_eq.item(),
            calendar_type="noleap",
        )
        expected_eq = 12.0 * 3600  # 12 hours at equator

        assert torch.allclose(
            daylength_eq, torch.tensor([expected_eq], dtype=daylength_eq.dtype), atol=1e-3
        )
        
    def test_compute_gsi_components(self):
        """Test GSI component calculation"""
        # Test with realistic environmental data
        tmin = torch.tensor([0.0, 5.0, 10.0])
        vpd = torch.tensor([2000.0, 2000.0, 2000.0])  # Moderate VPD
        daylen = torch.tensor([45000.0, 45000.0, 45000.0])  # About 12.5 hours
        
        f_temp, f_vpd, f_daylen = compute_gsi_components(
            tmin, vpd, daylen
        )
        
        # Test temperature component behavior
        assert f_temp[0] > 0.0  # Above threshold
        assert f_temp[1] >= f_temp[0]  # Higher temp = higher factor
        assert torch.all(f_temp >= 0.0) and torch.all(f_temp <= 1.0)
        
        # Test all components are in valid range
        assert torch.all(f_vpd >= 0.0) and torch.all(f_vpd <= 1.0)
        assert torch.all(f_daylen >= 0.0) and torch.all(f_daylen <= 1.0)
        
    def test_compute_gsi(self):
        """Test complete GSI calculation"""
        # Test with known values
        tmin = torch.tensor([5.0])
        vpd = torch.tensor([2000.0])
        daylen = torch.tensor([45000.0])  # seconds
        
        gsi = compute_gsi(
            temperature_min=tmin,
            vapor_pressure_deficit=vpd,
            daylength=daylen,
        )
        
        # GSI should be between 0 and 1
        assert torch.all(gsi >= 0.0) and torch.all(gsi <= 1.0)
        
        # Test limiting conditions
        gsi_dry = compute_gsi(
            temperature_min=tmin,
            vapor_pressure_deficit=torch.tensor([5000.0]),  # Very high VPD
            daylength=daylen,
        )
        
        assert gsi_dry < gsi  # Dry conditions should reduce GSI
        
    def test_thermal_time_accumulation(self):
        """Test thermal time accumulation"""
        temperatures = torch.tensor([15.0, 18.0, 22.0, 20.0, 16.0], dtype=torch.float32)
        base_temp = 10.0
        optimal_temp = 25.0
        max_temp = 35.0

        thermal_time = thermal_time_accumulation(
            temperatures, base_temp, optimal_temp, max_temp, method="linear"
        )

        expected_daily = torch.tensor([5.0, 8.0, 12.0, 10.0, 6.0], dtype=thermal_time.dtype)

        assert torch.allclose(thermal_time, expected_daily, atol=1e-6)
        
    def test_rolling_average(self):
        """Test rolling average calculation"""
        data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float32)
        window_size = 3

        rolling_avg = rolling_average(data, window_size)
        
        # Check that rolling average produces reasonable smoothed values
        # The actual function uses padding, so values will be different
        assert rolling_avg.shape == data.shape
        
        # Middle values should be smoothed versions of input
        # For window_size=3, check the middle region
        expected_middle = (data[2] + data[3] + data[4]) / 3.0  # For index 3
        assert torch.allclose(
            rolling_avg[3], expected_middle.to(rolling_avg.dtype), atol=1e-4
        )
        
    def test_julian_day_conversions(self):
        """Test Julian day date conversions"""
        # Test known dates
        doy = 120  # April 30 in non-leap year
        year = 2023
        
        date_obj = julian_day_to_date(year, doy)
        assert date_obj.month == 4
        assert date_obj.day == 30
        
        # Test reverse conversion
        doy_back = date_to_julian_day(date_obj)
        assert doy_back == doy
        
        # Test leap year
        doy_leap = 60  # February 29 in leap year
        year_leap = 2024
        
        date_leap = julian_day_to_date(year_leap, doy_leap)
        assert date_leap.month == 2
        assert date_leap.day == 29
        
    def test_compute_chilling_hours(self):
        """Test chilling hours calculation"""
        temperatures = torch.tensor([8.0, 2.0, -1.0, 5.0, 12.0, 15.0])
        
        chilling_hours = compute_chilling_hours(temperatures)
        
        # Only temps between 0-7掳C count as chilling hours
        expected = 3  # 2掳C, -1掳C, 5掳C count (8掳C is above threshold)
        total_chilling_hours = torch.sum(chilling_hours).item()
        assert total_chilling_hours == expected
        
    def test_validate_environmental_data(self):
        """Test environmental data validation"""
        # Test valid data
        valid_data = {
            'air_temperature': torch.tensor([15.0, 20.0]),
            'vapor_pressure_deficit': torch.tensor([1500.0, 2000.0]),
            'daylength': torch.tensor([43200.0, 45000.0])
        }
        
        required_vars = ['air_temperature', 'vapor_pressure_deficit', 'daylength']
        result = validate_environmental_data(valid_data, required_vars)
        assert result['valid'] is True
        
        # Test invalid temperature range
        invalid_data = {
            'air_temperature': torch.tensor([15.0, 100.0]),  # 100掳C is unrealistic
            'vapor_pressure_deficit': torch.tensor([1500.0, 2000.0])
        }
        
        result = validate_environmental_data(invalid_data, ['air_temperature'], check_ranges=True)
        assert result['valid'] is False or len(result['range_warnings']) > 0
            
        # Test missing required data
        incomplete_data = {
            'vapor_pressure_deficit': torch.tensor([1500.0, 2000.0])
        }
        
        result = validate_environmental_data(incomplete_data, ['air_temperature'])
        assert result['valid'] is False
        assert 'air_temperature' in result['missing_vars']


class TestPhenologyPhaseEnum:
    """Test phenology phase enumeration"""
    
    def test_phenology_phase_values(self):
        """Test phenology phase enumeration values"""
        assert PhenologyPhase.DORMANCY.value == "dormancy"
        assert PhenologyPhase.BUD_BREAK.value == "bud_break"
        assert PhenologyPhase.LEAF_EXPANSION.value == "leaf_expansion"
        assert PhenologyPhase.FULL_LEAF.value == "full_leaf"
        assert PhenologyPhase.SENESCENCE.value == "senescence"
        
    def test_phase_transitions(self):
        """Test logical phase transitions"""
        phases = list(PhenologyPhase)
        
        # Check that phases are in logical order
        assert phases[0] == PhenologyPhase.DORMANCY
        assert phases[-1] == PhenologyPhase.ABSCISSION
        
        # Check transitions make biological sense
        leaf_expansion_idx = phases.index(PhenologyPhase.LEAF_EXPANSION)
        full_leaf_idx = phases.index(PhenologyPhase.FULL_LEAF)
        assert leaf_expansion_idx < full_leaf_idx


class TestPhenologyOutput:
    """Test phenology output data structure"""
    
    def test_phenology_output_creation(self):
        """Test phenology output structure creation"""
        output = PhenologyOutput(
            phenology_phase=PhenologyPhase.FULL_LEAF,
            growth_factor=torch.tensor([0.8]),
            leaf_area_index=torch.tensor([4.5]),
            senescence_rate=torch.tensor([0.1])
        )
        
        assert output.phenology_phase == PhenologyPhase.FULL_LEAF
        assert torch.allclose(output.growth_factor, torch.tensor([0.8]))
        assert torch.allclose(output.leaf_area_index, torch.tensor([4.5]))
        assert torch.allclose(output.senescence_rate, torch.tensor([0.1]))
        
    def test_output_validation(self):
        """Test phenology output validation"""
        # Test valid ranges
        valid_output = PhenologyOutput(
            phenology_phase=PhenologyPhase.FULL_LEAF,
            growth_factor=torch.tensor([0.9]),
            leaf_area_index=torch.tensor([6.0]),
            senescence_rate=torch.tensor([0.05])
        )
        
        # Growth index should be 0-1
        assert torch.all(valid_output.growth_factor >= 0.0)
        assert torch.all(valid_output.growth_factor <= 1.0)
        
        # LAI should be positive
        assert torch.all(valid_output.leaf_area_index >= 0.0)
        
        # Senescence rate should be positive
        assert torch.all(valid_output.senescence_rate >= 0.0)


# Integration test with mock PhenologyBase
class MockPhenology(PhenologyBase):
    """Mock phenology class for testing abstract base"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__('mock_phenology', config)
        
    def compute_phenological_state(self, state, *, dt):
        """Mock implementation"""
        del state, dt
        return PhenologyOutput(
            phenology_phase=PhenologyPhase.LEAF_EXPANSION,
            growth_factor=torch.tensor([0.8]),
            leaf_area_index=torch.tensor([4.0]),
            senescence_rate=torch.tensor([0.1])
        )
        
    def get_dependencies(self):
        """Mock dependencies"""
        return ['air_temperature', 'day_of_year']


class TestPhenologyBaseIntegration:
    """Test phenology base class integration"""

    def test_functor_builds_parameter_object_without_model_defaults(
        self,
        basic_phenology_config,
        monkeypatch,
    ):
        """Functor construction must bind every field through its own accessor."""
        def reject_model_default(*args, **kwargs):
            del args, kwargs
            raise AssertionError(
                "PhenologyParameters must not fetch Functor defaults globally"
            )

        monkeypatch.setattr(
            "relic.scientific.biosphere.phenology.parameters."
            "get_global_manager",
            reject_model_default,
        )

        mock_pheno = MockPhenology(basic_phenology_config)

        from relic.scientific.biosphere.phenology.parameters import phenology_field_names

        assert mock_pheno.phenology_params.model_fields_set == set(
            phenology_field_names(mock_pheno.phenology_params.phenology_type)
        )

    def test_mock_phenology_creation(self, basic_phenology_config):
        """Test mock phenology creation and basic functionality"""
        mock_pheno = MockPhenology(basic_phenology_config)
        
        assert mock_pheno.metadata.name == 'mock_phenology'
        assert hasattr(mock_pheno, 'forward')
        assert hasattr(mock_pheno, 'forward')
        
        # Test dependencies
        deps = mock_pheno.get_dependencies()
        assert 'air_temperature' in deps
        assert 'day_of_year' in deps

    def test_forward_does_not_cache_scientific_state(
        self,
        basic_phenology_config,
    ):
        """StateManager, not the abstract base, owns phenology history."""
        mock_pheno = MockPhenology(basic_phenology_config)
        state = {
            "day_of_year": torch.tensor(180.0),
            "air_temperature": torch.tensor(15.0),
            "daylength": torch.tensor(43200.0),
            "leaf_carbon": torch.tensor(1.0),
            "froot_carbon": torch.tensor(2.0),
            "leaf_nitrogen": torch.tensor(0.04),
            "froot_nitrogen": torch.tensor(0.02),
        }

        mock_pheno(state, time=0.0, dt=3600.0)

        for hidden_state in (
            "current_phase",
            "current_lai",
            "accumulated_gdd",
            "accumulated_chilling",
            "last_gsi_values",
            "leaf_biomass",
            "root_biomass",
            "temperature_history",
            "daylength_history",
        ):
            assert not hasattr(mock_pheno, hidden_state)
        for dead_surface in (
            "_update_internal_state",
            "get_phenological_phase",
            "get_lai",
            "reset_phenological_state",
            "get_model_metadata",
        ):
            assert not hasattr(mock_pheno, dead_surface)

        init_source = inspect.getsource(PhenologyBase.__init__)
        assert "self._execution_count" not in init_source

    def test_canonical_call_counts_one_execution(
        self,
        basic_phenology_config,
    ):
        """FunctorBase is the single owner of lifecycle statistics."""
        mock_pheno = MockPhenology(basic_phenology_config)
        state = {
            "day_of_year": torch.tensor(180.0),
            "air_temperature": torch.tensor(15.0),
            "leaf_carbon": torch.tensor(1.0),
            "froot_carbon": torch.tensor(2.0),
            "leaf_nitrogen": torch.tensor(0.04),
            "froot_nitrogen": torch.tensor(0.02),
        }

        initial_stats = mock_pheno.get_execution_stats()
        assert set(initial_stats) == {
            "execution_count",
            "total_execution_time",
            "last_execution_time",
            "min_execution_time",
            "max_execution_time",
            "mean_execution_time",
            "error_count",
            "state",
        }
        assert initial_stats["execution_count"] == 0
        mock_pheno(state, time=0.0, dt=3600.0)
        assert mock_pheno.get_execution_stats()["execution_count"] == 1
        
    def test_forward_uses_explicit_cn_turnover_ledger(
        self,
        basic_phenology_config,
    ):
        """The base implementation must expose one exact, routed C/N ledger."""
        mock_pheno = MockPhenology(basic_phenology_config)
        dt = 3600.0
        state = {
            "day_of_year": torch.tensor(180.0),
            "air_temperature": torch.tensor(15.0),
            "leaf_carbon": torch.tensor(1.0),
            "froot_carbon": torch.tensor(2.0),
            "leaf_nitrogen": torch.tensor(0.04),
            "froot_nitrogen": torch.tensor(0.02),
        }

        updates = mock_pheno(
            state,
            time=0.0,
            dt=dt,
        )

        expected_keys = {
            "leaf_area_index",
            "phenological_factor",
            "phenology_senescence_factor",
            "phenology_phase",
            "leaf_carbon",
            "froot_carbon",
            "leaf_nitrogen",
            "froot_nitrogen",
            "plant_nitrogen_pool",
            "leaf_litterfall_carbon",
            "froot_litterfall_carbon",
            "leaf_litterfall_nitrogen",
            "froot_litterfall_nitrogen",
            "nitrogen_retranslocation",
            "diagnostics.phenology.carbon_turnover_balance_residual",
            "diagnostics.phenology.nitrogen_turnover_balance_residual",
        }
        assert set(updates) == expected_keys
        assert set(mock_pheno.metadata.output_variables) == expected_keys
        assert "growth_active" not in updates
        assert "senescence_rate" not in updates
        assert "leaf_turnover_flux" not in updates
        assert "froot_turnover_flux" not in updates
        assert "thermal_time" not in mock_pheno.OUTPUT_UNITS
        assert "growing_season_index" not in mock_pheno.OUTPUT_UNITS
        assert "harvest_index" not in mock_pheno.OUTPUT_UNITS
        rate_source = inspect.getsource(PhenologyBase._compute_rate_core)
        assert 'outputs["thermal_time"]' not in rate_source
        assert 'outputs["growing_season_index"]' not in rate_source

        state_keys = {
            "leaf_carbon",
            "froot_carbon",
            "leaf_nitrogen",
            "froot_nitrogen",
            "plant_nitrogen_pool",
        }
        for key in state_keys:
            assert isinstance(updates[key], UpdateItem)
            assert updates[key].mode is UpdateMode.ACCUMULATE
            assert updates[key].per_timestep is True

        flux_keys = {
            "leaf_litterfall_carbon",
            "froot_litterfall_carbon",
            "leaf_litterfall_nitrogen",
            "froot_litterfall_nitrogen",
            "nitrogen_retranslocation",
        }
        for key in flux_keys:
            assert isinstance(updates[key], UpdateItem)
            assert updates[key].mode is UpdateMode.REPLACE
            assert updates[key].per_timestep is False

        for key in (
            "leaf_litterfall_carbon",
            "froot_litterfall_carbon",
        ):
            assert updates[key].element == "C"
        for key in (
            "leaf_litterfall_nitrogen",
            "froot_litterfall_nitrogen",
        ):
            assert updates[key].element == "N"
        assert updates["nitrogen_retranslocation"].element is None

        leaf_rate = mock_pheno.phenology_params.leaf_turnover_rate * 0.1
        root_rate = mock_pheno.phenology_params.froot_turnover_rate
        expected_leaf_c_flux = state["leaf_carbon"] * leaf_rate
        expected_root_c_flux = state["froot_carbon"] * root_rate
        expected_leaf_n_turnover = state["leaf_nitrogen"] * leaf_rate
        expected_root_n_flux = state["froot_nitrogen"] * root_rate
        expected_retrans = (
            expected_leaf_n_turnover * mock_pheno.nitrogen_retrans_efficiency
        )
        expected_leaf_n_litter = expected_leaf_n_turnover - expected_retrans

        assert torch.allclose(
            updates["leaf_litterfall_carbon"].value,
            expected_leaf_c_flux,
        )
        assert torch.allclose(
            updates["froot_litterfall_carbon"].value,
            expected_root_c_flux,
        )
        assert torch.allclose(
            updates["nitrogen_retranslocation"].value,
            expected_retrans,
        )
        assert torch.allclose(
            updates["leaf_litterfall_nitrogen"].value,
            expected_leaf_n_litter,
        )
        assert torch.allclose(
            updates["froot_litterfall_nitrogen"].value,
            expected_root_n_flux,
        )

        carbon_residual = (
            updates["leaf_carbon"].value
            + updates["froot_carbon"].value
            + dt
            * (
                updates["leaf_litterfall_carbon"].value
                + updates["froot_litterfall_carbon"].value
            )
        )
        nitrogen_residual = (
            updates["leaf_nitrogen"].value
            + updates["froot_nitrogen"].value
            + updates["plant_nitrogen_pool"].value
            + dt
            * (
                updates["leaf_litterfall_nitrogen"].value
                + updates["froot_litterfall_nitrogen"].value
            )
        )
        assert torch.allclose(carbon_residual, torch.zeros_like(carbon_residual))
        assert torch.allclose(
            nitrogen_residual,
            torch.zeros_like(nitrogen_residual),
        )
        assert torch.equal(
            updates[
                "diagnostics.phenology.carbon_turnover_balance_residual"
            ].value,
            carbon_residual,
        )
        assert torch.equal(
            updates[
                "diagnostics.phenology.nitrogen_turnover_balance_residual"
            ].value,
            nitrogen_residual,
        )
        assert updates[
            "diagnostics.phenology.carbon_turnover_balance_residual"
        ].element is None
        assert updates[
            "diagnostics.phenology.nitrogen_turnover_balance_residual"
        ].element is None


class TestModelParameterScope:
    """A phenology model reads and declares only the parameters of its own model."""

    def test_each_model_reads_only_its_own_parameter_keys(self, monkeypatch):
        from relic.scientific.biosphere.phenology.models import (
            CropPhenology,
            GDDPhenology,
            GSIPhenology,
            StaticPhenology,
        )
        from relic.scientific.biosphere.phenology.parameters import (
            PHENOLOGY_MODEL_FIELDS,
            PHENOLOGY_PARAMETER_FIELD_KEYS,
        )

        read_keys = []
        original = PhenologyBase.get_parameter_value

        def recording(self, name, **kwargs):
            namespace = kwargs.get("namespace")
            read_keys.append(f"{namespace}.{name}" if namespace else name)
            return original(self, name, **kwargs)

        monkeypatch.setattr(PhenologyBase, "get_parameter_value", recording)
        model_keys = {
            kind: {PHENOLOGY_PARAMETER_FIELD_KEYS[field] for field in fields}
            for kind, fields in PHENOLOGY_MODEL_FIELDS.items()
        }
        for model_class in (StaticPhenology, GDDPhenology, GSIPhenology, CropPhenology):
            read_keys.clear()
            kind = model_class.PHENOLOGY_TYPE
            model = model_class(name=f"scope_{kind}")
            foreign = set().union(*(keys for other, keys in model_keys.items() if other != kind))
            foreign -= model_keys[kind]
            assert model_keys[kind] <= set(read_keys), kind
            assert not foreign & set(read_keys), kind
            assert model_keys[kind] <= set(model.metadata.parameters), kind
            assert not foreign & set(model.metadata.parameters), kind


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


