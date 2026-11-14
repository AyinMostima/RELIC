"""
L1 Unit Tests for Phenology Models

Tests individual phenology model functionality with known inputs/outputs.
Validates parameter loading, conservation checks, and mathematical correctness.
"""

import inspect

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../'))

from relic.scientific.biosphere.phenology.models import (
    StaticPhenology, GDDPhenology, GSIPhenology, CropPhenology
)
from relic.scientific.biosphere.phenology.base import PhenologyPhase
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant

@pytest.fixture(scope="module", autouse=True)
def _load_relic_parameter_database_once():
    """Load the parameter database once for this L1 test module; the
    process-wide manager is restored exactly after the module."""

    param_manager = get_global_manager()
    captured = param_manager.snapshot()
    param_manager.load_from_database('relic/parameters/database')
    yield
    param_manager.restore(captured)


@pytest.mark.parametrize(
    ("model_class", "expected_type"),
    [
        (StaticPhenology, "static"),
        (GDDPhenology, "gdd"),
        (GSIPhenology, "gsi"),
        (CropPhenology, "crop"),
    ],
)
def test_concrete_model_parameters_use_functor_accessor_only(
    model_class,
    expected_type,
    monkeypatch,
):
    """Concrete Functors must not trigger parameter-model global defaults."""
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

    model = model_class()

    from relic.scientific.biosphere.phenology.parameters import phenology_field_names

    assert model.phenology_params.phenology_type == expected_type
    assert model.phenology_params.model_fields_set == set(
        phenology_field_names(expected_type)
    )
    if expected_type == "crop":
        assert model.phenology_params.crop_type == "wheat"


class TestStaticPhenology:
    """Test StaticPhenology model with calendar-based transitions."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Create model instance
        self.model = StaticPhenology()
        
    def test_instantiation(self):
        """Test model instantiation."""
        assert self.model is not None
        assert self.model.metadata.name == "static_phenology"
        
    def test_dormancy_phase(self):
        """Test dormancy phase detection."""
        # Winter day
        day_of_year = torch.tensor(30.0)  # January 30
        temperature = torch.tensor(273.15)  # 0degC in Kelvin
        environment = {
            'temperature_min': torch.tensor(268.15),  # -5degC
            'temperature_max': torch.tensor(278.15),  # 5degC
        }
        
        output = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output.phenology_phase == PhenologyPhase.DORMANCY
        assert output.growth_factor.item() < 0.1
        
    def test_growing_phase(self):
        """Test growing season phase."""
        # Summer day
        day_of_year = torch.tensor(180.0)  # June 29
        temperature = torch.tensor(293.15)  # 20degC in Kelvin
        environment = {
            'temperature_min': torch.tensor(288.15),  # 15degC
            'temperature_max': torch.tensor(298.15),  # 25degC
        }
        
        output = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output.phenology_phase == PhenologyPhase.FULL_LEAF
        assert output.growth_factor.item() > 0.9
        
    def test_conservation_checks(self):
        """Test conservation law enforcement."""
        # Normal conditions should pass conservation
        day_of_year = torch.tensor(150.0)
        temperature = torch.tensor(288.15)
        environment = {
            'temperature_min': torch.tensor(283.15),
            'temperature_max': torch.tensor(293.15),
        }
        
        output = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        # Check values are within bounds
        assert 0 <= output.growth_factor.item() <= 1.0
        assert output.leaf_area_index.item() >= 0
        

class TestGDDPhenology:
    """Test GDD-based phenology model."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.model = GDDPhenology()
        
    def test_gdd_accumulation(self):
        """Test GDD accumulation over time."""
        accumulated_gdd = torch.tensor(0.0)
        previous_day = torch.tensor(89.0)
        seconds_per_day = torch.tensor(
            get_physical_constant("seconds_per_day")
        )

        # Simulate early season
        for day in range(90, 150, 10):
            day_of_year = torch.tensor(float(day))
            temperature = torch.tensor(283.15 + day/10)  # Gradual warming
            environment = {
                'air_temperature_min': torch.tensor(278.15 + day/10),
                'air_temperature_max': torch.tensor(288.15 + day/10),
                'phenology_gdd_accumulated': accumulated_gdd,
                'phenology_gdd_previous_day_of_year': previous_day,
            }
            
            output = self.model.compute_phenological_state(
                {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
                dt=seconds_per_day,
            )
            
            # GDD should increase over time
            if day > 90:
                assert output.thermal_time.item() > 0
            accumulated_gdd = output.additional_outputs[
                'phenology_gdd_accumulated'
            ]
            previous_day = output.additional_outputs[
                'phenology_gdd_previous_day_of_year'
            ]
                
    def test_reset_at_year_boundary(self):
        """Test GDD reset at year boundary."""
        # Late in year
        day_of_year = torch.tensor(360.0)
        temperature = torch.tensor(273.15)
        seconds_per_day = torch.tensor(
            get_physical_constant("seconds_per_day")
        )
        environment = {
            'air_temperature_min': torch.tensor(268.15),
            'air_temperature_max': torch.tensor(278.15),
            'phenology_gdd_accumulated': torch.tensor(75.0),
            'phenology_gdd_previous_day_of_year': torch.tensor(359.0),
        }
        
        late_year = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=seconds_per_day,
        )
        
        # Early next year - should reset
        day_of_year = torch.tensor(5.0)
        environment['phenology_gdd_accumulated'] = late_year.additional_outputs[
            'phenology_gdd_accumulated'
        ]
        environment['phenology_gdd_previous_day_of_year'] = (
            late_year.additional_outputs['phenology_gdd_previous_day_of_year']
        )
        early_year = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=seconds_per_day,
        )
        
        # The canonical state resets; no Functor-owned accumulator is used.
        assert early_year.additional_outputs[
            'phenology_gdd_accumulated'
        ].item() == pytest.approx(0.0)
        assert early_year.additional_outputs[
            'phenology_gdd_previous_day_of_year'
        ].item() == pytest.approx(5.0)
        assert not hasattr(self.model, 'accumulated_gdd')

    def test_required_temperature_extrema_are_canonical_and_fail_fast(self):
        """Require canonical extrema without aliases or estimated values."""
        with pytest.raises(ValueError, match="air_temperature_max"):
            self.model.compute_phenological_state(
                {
                    "temperature_min": torch.tensor(283.15),
                    "temperature_max": torch.tensor(293.15),
                    "day_of_year": torch.tensor(120.0),
                    "air_temperature": torch.tensor(288.15),
                },
                dt=get_physical_constant("seconds_per_day"),
            )

    def test_metadata_uses_model_specific_gdd_diagnostic(self):
        """Keep GDD thermal time distinct from crop accumulated PHU."""
        outputs = set(self.model.metadata.output_variables)
        inputs = set(self.model.metadata.input_variables)
        assert "phenology_gdd_accumulated" in inputs
        assert "phenology_gdd_previous_day_of_year" in inputs
        assert "phenology_gdd_accumulated" in outputs
        assert "phenology_gdd_previous_day_of_year" in outputs
        assert "diagnostics.phenology.accumulated_gdd" in outputs
        assert "thermal_time" not in outputs
        
        
class TestGSIPhenology:
    """Test GSI-based phenology model."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.model = GSIPhenology()

    def _environment(self, **overrides):
        """Build the canonical daily forcing and temporal-state contract."""
        environment = {
            "air_temperature_min": torch.tensor(283.15),
            "vapor_pressure_deficit": torch.tensor(1000.0),
            "daylength": torch.tensor(14 * 3600.0),
            "phenology_gsi_history": torch.zeros(self.model.memory_days),
            "phenology_gsi_history_index": torch.tensor(0.0),
            "phenology_gsi_history_count": torch.tensor(0.0),
            "phenology_gsi_in_growing_season": torch.tensor(0.0),
        }
        environment.update(overrides)
        return environment
        
    def test_gsi_calculation(self):
        """Test GSI calculation with environmental drivers."""
        day_of_year = torch.tensor(150.0)
        temperature = torch.tensor(288.15)  # 15degC
        
        environment = self._environment()
        
        output = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        # Should have positive GSI
        assert output.growing_season_index is not None
        assert output.growing_season_index.item() > 0.5
        
    def test_stress_conditions(self):
        """Test GSI under stress conditions."""
        day_of_year = torch.tensor(150.0)
        temperature = torch.tensor(288.15)
        
        environment = self._environment(
            air_temperature_min=torch.tensor(268.15),
            vapor_pressure_deficit=torch.tensor(4000.0),
            daylength=torch.tensor(8 * 3600.0),
        )
        
        output = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        # Should have low GSI due to stress
        assert output.growing_season_index.item() < 0.3

    @pytest.mark.parametrize(
        "missing_key",
        [
            "air_temperature_min",
            "vapor_pressure_deficit",
            "daylength",
            "phenology_gsi_history",
            "phenology_gsi_history_index",
            "phenology_gsi_history_count",
            "phenology_gsi_in_growing_season",
        ],
    )
    def test_required_environmental_inputs_fail_fast(self, missing_key):
        """Reject incomplete GSI forcing instead of synthesizing defaults."""
        environment = self._environment()
        del environment[missing_key]
        with pytest.raises(ValueError, match=missing_key):
            self.model.compute_phenological_state(
                {**environment, "day_of_year": torch.tensor(150.0), "air_temperature": torch.tensor(288.15)},
                dt=get_physical_constant("seconds_per_day"),
            )

    def test_metadata_namespaces_growing_season_index(self):
        """Expose GSI only as a namespaced observation diagnostic."""
        inputs = set(self.model.metadata.input_variables)
        outputs = set(self.model.metadata.output_variables)
        assert "diagnostics.phenology.growing_season_index" in outputs
        assert "growing_season_index" not in outputs
        assert "air_temperature_min" in inputs
        assert "temperature_min" not in inputs
        assert "soil_water_potential" not in inputs
        for key in (
            "phenology_gsi_history",
            "phenology_gsi_history_index",
            "phenology_gsi_history_count",
            "phenology_gsi_in_growing_season",
        ):
            assert key in inputs
            assert key in outputs
        

class TestCropPhenology:
    """Test crop-specific phenology model."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.model = CropPhenology()

    @staticmethod
    def temporal_environment(
        *,
        accumulated_phu=0.0,
        is_sown=False,
        is_harvested=False,
        vernalization_days=0.0,
    ):
        """Build the canonical StateManager-owned Crop temporal state."""
        return {
            'phenology_crop_accumulated_phu': torch.tensor(accumulated_phu),
            'phenology_crop_is_sown': torch.tensor(float(is_sown)),
            'phenology_crop_is_harvested': torch.tensor(float(is_harvested)),
            'phenology_crop_vernalization_days': torch.tensor(
                vernalization_days
            ),
        }
        
    def test_crop_development_stages(self):
        """Test progression through crop development stages."""
        # Sowing day
        day_of_year = torch.tensor(120.0)  # Default sowing day
        temperature = torch.tensor(288.15)
        environment = self.temporal_environment()
        
        output = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output.additional_outputs['phenology_crop_is_sown'].item() == 1.0
        assert output.additional_outputs[
            'phenology_crop_is_harvested'
        ].item() == 0.0
        assert output.additional_outputs[
            'phenology_crop_accumulated_phu'
        ].item() == pytest.approx(10.0)
        
    def test_harvest_index_calculation(self):
        """Test harvest index calculation during grain filling."""
        # One daily step adds 10 PHU, producing exactly DI = 840 / 1200.
        day_of_year = torch.tensor(250.0)
        temperature = torch.tensor(288.15)
        environment = self.temporal_environment(
            accumulated_phu=830.0,
            is_sown=True,
        )

        output = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )

        assert output.additional_outputs[
            'phenology_crop_accumulated_phu'
        ].item() == pytest.approx(840.0)
        assert output.additional_outputs['harvest_index'] > 0
        
    def test_crop_does_not_own_biomass_allocation_formula(self):
        """Keep biomass allocation with the selected BiomassAllocation owner."""
        day_of_year = torch.tensor(150.0)
        temperature = torch.tensor(
            get_physical_constant('T_FREEZE')
            + self.model.phenology_params.gdd_base_temp
        )
        environment = self.temporal_environment(
            accumulated_phu=360.0,
            is_sown=True,
        )
        
        output = self.model.compute_phenological_state(
            {**environment, "day_of_year": day_of_year, "air_temperature": temperature},
            dt=get_physical_constant("seconds_per_day"),
        )
        
        assert output.allocation_fractions == {}
        assert not hasattr(self.model, '_compute_allocation_fractions')

    def test_metadata_does_not_claim_unused_forcing(self):
        """Exclude forcing ports that are not consumed by the crop law."""
        assert "soil_water_potential" not in self.model.metadata.input_variables
        assert "nitrogen_availability" not in self.model.metadata.input_variables

    def test_metadata_uses_model_specific_phu_diagnostic(self):
        """Keep crop accumulated PHU distinct from GDD thermal time."""
        outputs = set(self.model.metadata.output_variables)
        assert "diagnostics.phenology.accumulated_phu" in outputs
        assert "thermal_time" not in outputs
        assert self.model.OUTPUT_UNITS["harvest_index"] == "dimensionless"


class TestParameterLoading:
    """Test parameter loading without defaults."""
    
    def test_no_default_parameters(self):
        """Verify no default parameters are used."""
        # Should not raise any errors about missing parameters
        static = StaticPhenology()
        gdd = GDDPhenology()
        gsi = GSIPhenology()
        crop = CropPhenology()

        # All models should instantiate successfully
        assert static is not None
        assert gdd is not None
        assert gsi is not None
        assert crop is not None

    def test_models_use_canonical_functor_parameter_entrypoint(self):
        """Prevent bridge convenience access from becoming a second API."""
        source = inspect.getsource(__import__(
            "relic.scientific.biosphere.phenology.models",
            fromlist=["models"],
        ))
        assert "self.param(" not in source
        

class TestConservationEnforcement:
    """Test conservation law enforcement."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        pass
        
    def test_negative_lai_rejected(self):
        """Test that negative LAI values are rejected."""
        from relic.scientific.biosphere.phenology.base import PhenologyOutput
        
        model = StaticPhenology()
        
        # Create output with negative LAI
        bad_output = PhenologyOutput(
            phenology_phase=PhenologyPhase.FULL_LEAF,
            growing_season_index=torch.tensor(1.0),
            thermal_time=torch.tensor(100.0),
            leaf_area_index=torch.tensor(-1.0),  # Invalid negative LAI
            growth_factor=torch.tensor(0.5),
            senescence_rate=torch.tensor(0.1),
        )
        
        # Should raise bounds validation error
        with pytest.raises(ValueError):
            model._validate_output_bounds(bad_output)
            
    def test_allocation_sum_validated(self):
        """Test allocation fractions must sum to 1."""
        from relic.scientific.biosphere.phenology.base import PhenologyOutput
        
        model = StaticPhenology()
        
        # Create output with invalid allocation sum
        bad_output = PhenologyOutput(
            phenology_phase=PhenologyPhase.FULL_LEAF,
            growing_season_index=torch.tensor(1.0),
            thermal_time=torch.tensor(100.0),
            leaf_area_index=torch.tensor(5.0),
            growth_factor=torch.tensor(0.5),
            senescence_rate=torch.tensor(0.1),
            allocation_fractions={
                'leaf': torch.tensor(0.4),
                'stem': torch.tensor(0.3),
                'root': torch.tensor(0.2),  # Sum = 0.9, not 1.0
            }
        )
        
        # Should raise bounds validation error
        with pytest.raises(ValueError):
            model._validate_output_bounds(bad_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

