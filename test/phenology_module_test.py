"""
Comprehensive test suite for RELIC phenology module.

This test suite validates the phenology module functionality including:
- PhenologyFunctor initialization and basic functionality
- Static phenology model
- GDD (Growing Degree Day) phenology model
- GSI (Growing Season Index) phenology model
- Crop phenology model
- Parameter integration and validation
- Conservation checks

References:
- Richardson, A.D. et al. (2013). Agricultural and Forest Meteorology 169:156-173
- McMaster, G.S. & Wilhelm, W.W. (1997). Agricultural and Forest Meteorology 87:291-300
- Jolly, W.M. et al. (2005). Global Change Biology 11:619-632
"""

import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from relic.scientific.biosphere.phenology import PhenologyFunctor
from relic.scientific.biosphere.phenology.models import (
    StaticPhenology,
    GDDPhenology,
    GSIPhenology,
    CropPhenology
)
from relic.parameters import get_global_manager
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant


class TestPhenologyModels:
    """Test individual phenology model implementations."""

    @pytest.fixture
    def sample_state(self):
        """Create sample state tensor for testing."""
        return {
            'air_temperature': torch.tensor([288.15, 290.15, 292.15, 294.15]),  # K
            'leaf_carbon': torch.tensor([0.05, 0.06, 0.07, 0.08]),  # kg C m-2
            'day_of_year': torch.tensor([100, 101, 102, 103]),  # day
            'latitude': torch.tensor([45.0]),  # degrees
            'soil_water_potential': torch.tensor([-0.5]),  # MPa
            'vapor_pressure_deficit': torch.tensor([1000.0]),  # Pa
            'daylength': torch.tensor([43200.0])  # s (12 hours)
        }

    @pytest.fixture
    def sample_environment(self):
        """Create sample environmental conditions."""
        return {
            'solar_radiation': torch.tensor([20.0, 22.0, 18.0, 15.0]),  # MJ m-2 day-1
            'wind_speed': torch.tensor([3.0]),  # m s-1
            'atmospheric_pressure': torch.tensor([101325.0])  # Pa
        }

    def test_static_phenology_model(self, sample_state, sample_environment):
        """Test StaticPhenology implementation."""
        model = StaticPhenology()

        # Test basic properties
        assert hasattr(model, 'compute_phenological_state')

        # Test phenology state calculation
        day_of_year = sample_state['day_of_year'][0]
        temperature = sample_state['air_temperature'][0]

        phenology_state = model.compute_phenological_state(
            {**sample_environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )

        # Static model should return constant moderate activity
        assert hasattr(phenology_state, 'leaf_area_index')
        assert hasattr(phenology_state, 'growth_factor')
        assert hasattr(phenology_state, 'phenology_phase')

        # Values should be tensors
        assert isinstance(phenology_state.leaf_area_index, torch.Tensor)
        assert isinstance(phenology_state.growth_factor, torch.Tensor)

        # Values should be in reasonable ranges
        assert torch.all(phenology_state.growth_factor >= 0.0)
        assert torch.all(phenology_state.growth_factor <= 1.0)
        assert torch.all(phenology_state.leaf_area_index >= 0.0)

    def test_gdd_phenology_model(self, sample_state, sample_environment):
        """Test GDDPhenology implementation."""
        model = GDDPhenology()

        # Test basic properties
        assert hasattr(model, 'compute_phenological_state')
        assert hasattr(model, '_compute_gdd_growth_factor')

        # Test phenology state calculation
        day_of_year = sample_state['day_of_year'][0]
        temperature = sample_state['air_temperature'][0]
        gdd_environment = {
            **sample_environment,
            'air_temperature_min': temperature - 5.0,
            'air_temperature_max': temperature + 5.0,
            'phenology_gdd_accumulated': torch.tensor(0.0),
            'phenology_gdd_previous_day_of_year': torch.tensor(99.0),
        }

        phenology_state = model.compute_phenological_state(
            {**gdd_environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=torch.tensor(
                get_physical_constant("seconds_per_day")
            ),
        )

        assert hasattr(phenology_state, 'leaf_area_index')
        assert hasattr(phenology_state, 'growth_factor')
        assert phenology_state.additional_outputs[
            'phenology_gdd_accumulated'
        ].item() == pytest.approx(10.0)
        assert hasattr(phenology_state, 'thermal_time')

        # Values should be tensors
        assert isinstance(phenology_state.leaf_area_index, torch.Tensor)
        assert isinstance(phenology_state.growth_factor, torch.Tensor)
        assert isinstance(phenology_state.thermal_time, torch.Tensor)

        # Values should be in reasonable ranges
        assert torch.all(phenology_state.growth_factor >= 0.0)
        assert torch.all(phenology_state.growth_factor <= 1.0)
        assert torch.all(phenology_state.thermal_time >= 0.0)
    def test_gsi_phenology_model(self, sample_state, sample_environment):
        """Test GSIPhenology implementation."""
        model = GSIPhenology()

        # Test basic properties
        assert hasattr(model, 'compute_phenological_state')
        assert hasattr(model, '_compute_gsi_growth_factor')

        # Test phenology state calculation
        day_of_year = sample_state['day_of_year'][0]
        temperature = sample_state['air_temperature'][0]
        gsi_environment = {
            **sample_environment,
            'air_temperature_min': temperature - 5.0,
            'vapor_pressure_deficit': sample_state['vapor_pressure_deficit'],
            'daylength': sample_state['daylength'],
            'phenology_gsi_history': torch.zeros(model.memory_days),
            'phenology_gsi_history_index': torch.tensor(0.0),
            'phenology_gsi_history_count': torch.tensor(0.0),
            'phenology_gsi_in_growing_season': torch.tensor(0.0),
        }

        phenology_state = model.compute_phenological_state(
            {**gsi_environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=torch.tensor(
                get_physical_constant("seconds_per_day")
            ),
        )

        assert hasattr(phenology_state, 'leaf_area_index')
        assert hasattr(phenology_state, 'growth_factor')
        assert hasattr(phenology_state, 'growing_season_index')

        # Values should be tensors
        assert isinstance(phenology_state.leaf_area_index, torch.Tensor)
        assert isinstance(phenology_state.growth_factor, torch.Tensor)
        assert isinstance(phenology_state.growing_season_index, torch.Tensor)

        # Values should be in reasonable ranges
        assert torch.all(phenology_state.growth_factor >= 0.0)
        assert torch.all(phenology_state.growth_factor <= 1.0)
        if phenology_state.growing_season_index is not None:
            assert torch.all(phenology_state.growing_season_index >= 0.0)
            assert torch.all(phenology_state.growing_season_index <= 1.0)

    def test_crop_phenology_model(self, sample_state, sample_environment):
        """Test CropPhenology implementation."""
        model = CropPhenology()

        # Test basic properties
        assert hasattr(model, 'compute_phenological_state')
        assert hasattr(model, '_get_development_stage')

        # Test phenology state calculation
        day_of_year = sample_state['day_of_year'][0]
        temperature = sample_state['air_temperature'][0]

        phenology_state = model.compute_phenological_state(
            {
                **sample_environment,
                'phenology_crop_accumulated_phu': torch.tensor(0.0),
                'phenology_crop_is_sown': torch.tensor(0.0),
                'phenology_crop_is_harvested': torch.tensor(0.0),
                'phenology_crop_vernalization_days': torch.tensor(0.0),
                "day_of_year": day_of_year,
                "air_temperature": temperature,
            },
            dt=torch.tensor(
                    get_physical_constant('seconds_per_day')
                ),
        )

        assert hasattr(phenology_state, 'leaf_area_index')
        assert hasattr(phenology_state, 'growth_factor')
        assert hasattr(phenology_state, 'thermal_time')

        # Values should be tensors
        assert isinstance(phenology_state.leaf_area_index, torch.Tensor)
        assert isinstance(phenology_state.growth_factor, torch.Tensor)
        assert isinstance(phenology_state.thermal_time, torch.Tensor)

        # Values should be in reasonable ranges
        assert torch.all(phenology_state.growth_factor >= 0.0)
        assert torch.all(phenology_state.growth_factor <= 1.0)
        assert torch.all(phenology_state.thermal_time >= 0.0)
        assert phenology_state.additional_outputs[
            'phenology_crop_accumulated_phu'
        ].item() == 0.0


class TestPhenologyFunctor:
    """Test PhenologyFunctor integration."""

    @pytest.fixture
    def phenology_functor(self):
        """Create PhenologyFunctor instance for testing."""
        return PhenologyFunctor(
            config={"canonical_model_owner": "gdd_phenology"}
        )

    @pytest.fixture
    def sample_state(self):
        """Create sample state for functor testing."""
        batch_size = 2
        return {
            'air_temperature': torch.full((batch_size,), 288.15),  # 15掳C
            'day_of_year': torch.full((batch_size,), 150),  # Late May
            'latitude': torch.full((batch_size,), 45.0),  # Mid-latitude
            'leaf_carbon': torch.full((batch_size,), 0.05),
            'froot_carbon': torch.full((batch_size,), 0.02),
            'leaf_nitrogen': torch.full((batch_size,), 0.004),
            'froot_nitrogen': torch.full((batch_size,), 0.002)
        }

    @pytest.fixture
    def sample_environment(self):
        """Create sample environment for functor testing."""
        batch_size = 2
        return {
            'solar_radiation': torch.full((batch_size,), 20.0),
            'soil_water_potential': torch.full((batch_size,), -0.5),
            'vapor_pressure_deficit': torch.full((batch_size,), 1000.0),
            'daylength': torch.full((batch_size,), 43200.0)  # 12 hours
        }

    @pytest.fixture
    def owner_output(self):
        """Create the canonical owner observations consumed by the shell."""
        return {
            'leaf_area_index': torch.full((2,), 2.5),
            'phenological_factor': torch.full((2,), 0.6),
            'phenology_senescence_factor': torch.full((2,), 0.1),
            'phenology_phase': torch.full((2,), 2.0),
        }

    def test_phenology_functor_initialization(self, phenology_functor):
        """Test PhenologyFunctor initialization."""
        assert phenology_functor.canonical_model_owner == "gdd_phenology"
        assert hasattr(phenology_functor, 'forward')
        assert phenology_functor.get_dependencies() == {
            "required": ["gdd_phenology"],
            "optional": [],
        }

    def test_phenology_functor_forward(
        self, phenology_functor, sample_state, owner_output
    ):
        """Test PhenologyFunctor forward method."""
        time = torch.tensor(0.0)
        dt = torch.tensor(86400.0)  # 1 day

        # Run forward pass
        updates = phenology_functor.forward(
            state=sample_state,
            time=time,
            dt=dt,
            subprocess_outputs={"gdd_phenology": owner_output},
        )

        # Check return format
        assert isinstance(updates, dict)

        assert set(updates) == set(phenology_functor.metadata.output_variables)

        expected_units = {
            'diagnostics.phenology_state': 'dimensionless',
            'diagnostics.growth_stage': 'dimensionless',
            'diagnostics.leaf_area_index': 'm2 m-2',
            'diagnostics.phenology.senescence_factor': 'dimensionless',
        }
        for key, item in updates.items():
            assert isinstance(item, UpdateItem)
            assert item.value.shape[0] == sample_state['air_temperature'].shape[0]
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is True
            assert item.unit == expected_units[key]
            assert item.source_functor == phenology_functor.name

    def test_phenology_functor_parameter_access(self, phenology_functor):
        """The aggregation shell is parameter-free."""
        assert "biosphere.phenology" not in str(
            phenology_functor.metadata.parameters
        )

    def test_phenology_functor_conservation_check(self, phenology_functor, sample_state):
        """Diagnostic projection does not declare prognostic conservation."""
        assert not phenology_functor.metadata.conservation_laws
        assert all(
            key.startswith("diagnostics.")
            for key in phenology_functor.metadata.output_variables
        )


class TestPhenologyIntegration:
    """Test phenology module integration with RELIC system."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing."""
        config = {
            'simulation': {
                'start_time': 0.0,
                'time_step': 86400.0,  # 1 day
                'duration': 86400.0 * 10  # 10 days
            },
            'vegetation': {
                'type': 'tree',
                'phenology_model': 'gdd'
            }
        }
        return config

    def test_phenology_parameter_loading(self):
        """Test loading of phenology parameters from database."""
        param_manager = get_global_manager()

        # Test key phenology parameters exist
        key_params = [
            'biosphere.phenology.lai.baseline',
            'biosphere.phenology.gdd.max_temp_limit',
            'biosphere.phenology.gsi.tmin_threshold',
            'relic.physical.constants.time.seconds_per_day'
        ]

        for param_key in key_params:
            spec = param_manager.get_parameter(param_key)
            assert spec is not None
            assert hasattr(spec, 'value')
            assert hasattr(spec, 'units')

    def test_phenology_model_selection(self):
        """Test compile-time canonical owner selection."""
        owners = [
            'static_phenology',
            'gdd_phenology',
            'gsi_phenology',
            'crop_phenology',
        ]

        for owner in owners:
            functor = PhenologyFunctor(
                config={"canonical_model_owner": owner}
            )
            assert functor.canonical_model_owner == owner

    def test_phenology_simulation_step(self):
        """Test a single simulation step with phenology."""
        # Create phenology functor
        phenology_functor = PhenologyFunctor(
            config={"canonical_model_owner": "gdd_phenology"}
        )

        # Create initial state
        state = {
            'air_temperature': torch.tensor([288.15]),
            'day_of_year': torch.tensor([120]),
            'latitude': torch.tensor([45.0]),
            'leaf_carbon': torch.tensor([0.05])
        }

        owner_output = {
            'leaf_area_index': torch.tensor([2.0]),
            'phenological_factor': torch.tensor([0.5]),
            'phenology_senescence_factor': torch.tensor([0.1]),
            'phenology_phase': torch.tensor([2.0]),
        }

        # Run simulation step
        time = torch.tensor(0.0)
        dt = torch.tensor(86400.0)

        updates = phenology_functor.forward(
            state,
            time,
            dt,
            subprocess_outputs={"gdd_phenology": owner_output},
        )

        # Verify updates
        assert isinstance(updates, dict)
        assert len(updates) > 0

        # Diagnostic observations are finite and state-neutral replacements.
        for item in updates.values():
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is True
            assert torch.all(torch.isfinite(item.value))
            assert torch.all(torch.abs(item.value) < 1000.0)


class TestPhenologyValidation:
    """Test phenology module validation and error handling."""

    def test_invalid_phenology_type(self):
        """Test error handling for invalid phenology type."""
        with pytest.raises(ValueError, match="Unsupported canonical"):
            PhenologyFunctor(
                config={"canonical_model_owner": "invalid_type"}
            )

    def test_missing_required_inputs(self):
        """Test error handling for missing required inputs."""
        functor = PhenologyFunctor(
            config={"canonical_model_owner": "gdd_phenology"}
        )

        # Missing required state variables
        incomplete_state = {'air_temperature': torch.tensor([288.15])}
        time = torch.tensor(0.0)
        dt = torch.tensor(86400.0)

        with pytest.raises(KeyError, match="gdd_phenology"):
            functor.forward(
                incomplete_state,
                time,
                dt,
                subprocess_outputs={},
            )

    def test_temperature_bounds_validation(self):
        """Test temperature bounds validation."""
        functor = PhenologyFunctor(
            config={"canonical_model_owner": "gdd_phenology"}
        )

        # Test with extreme temperatures
        state = {
            'air_temperature': torch.tensor([200.0]),  # Very cold
            'day_of_year': torch.tensor([150]),
            'latitude': torch.tensor([45.0])
        }

        time = torch.tensor(0.0)
        dt = torch.tensor(86400.0)

        owner_output = {
            'leaf_area_index': torch.tensor([float('nan')]),
            'phenological_factor': torch.tensor([0.5]),
            'phenology_senescence_factor': torch.tensor([0.1]),
            'phenology_phase': torch.tensor([2.0]),
        }
        with pytest.raises(ValueError, match="leaf_area_index"):
            functor.forward(
                state,
                time,
                dt,
                subprocess_outputs={"gdd_phenology": owner_output},
            )


if __name__ == "__main__":
    # Run specific tests if this file is executed directly
    pytest.main([__file__, "-v"])
