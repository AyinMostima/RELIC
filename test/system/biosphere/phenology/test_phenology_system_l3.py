"""L3 System Tests for Phenology Module.

End-to-end tests for complete phenology workflow with parameter system.
"""

import pytest
import torch
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from relic.core.state import UpdateItem, UpdateMode
from relic.core.functor import ExecutionError
from relic.core.dependency_graph import DependencyGraphResolver
from relic.parameters import UnifiedParameterManager
from relic.utils.physical_constants import get_physical_constant
from relic.scientific.biosphere.phenology import (
    StaticPhenology as StaticPhenologyFunctor,
    GDDPhenology as GDDPhenologyFunctor,
    GSIPhenology as GSIPhenologyFunctor,
    CropPhenology as CropPhenologyFunctor,
    PhenologyFunctor,
)

SECONDS_PER_DAY = float(get_physical_constant("seconds_per_day"))

try:
    from relic.scientific.biosphere.phenology import PhenologyOrchestrator
except ImportError:  # pragma: no cover - historical builds without orchestrator
    PhenologyOrchestrator = None


def _canonical_plant_state() -> dict[str, torch.Tensor]:
    """Return required vegetation pools in canonical SI units."""
    return {
        'leaf_carbon': torch.tensor(0.20),
        'froot_carbon': torch.tensor(0.15),
        'leaf_nitrogen': torch.tensor(0.008),
        'froot_nitrogen': torch.tensor(0.004),
        'plant_nitrogen_pool': torch.tensor(0.020),
    }


def _canonical_gsi_state(functor) -> dict[str, torch.Tensor]:
    """Return StateManager-owned GSI temporal inputs."""
    return {
        'phenology_gsi_history': torch.zeros(functor.memory_days),
        'phenology_gsi_history_index': torch.tensor(0.0),
        'phenology_gsi_history_count': torch.tensor(0.0),
        'phenology_gsi_in_growing_season': torch.tensor(0.0),
    }


def _canonical_crop_state() -> dict[str, torch.Tensor]:
    """Return StateManager-owned Crop temporal inputs."""
    return {
        'phenology_crop_accumulated_phu': torch.tensor(0.0),
        'phenology_crop_is_sown': torch.tensor(1.0),
        'phenology_crop_is_harvested': torch.tensor(0.0),
        'phenology_crop_vernalization_days': torch.tensor(0.0),
    }


class TestPhenologySystemIntegration:
    """System-level tests for phenology module."""
    
    @pytest.fixture
    def phenology_db_root(self):
        """Create an isolated copy of the parameter database for testing."""
        import relic
        import tempfile

        source_db = Path(relic.__file__).parent / "parameters" / "database"
        temp_root = Path(tempfile.mkdtemp(prefix="phenology_db_"))
        target_db = temp_root / "database"
        shutil.copytree(source_db, target_db)
        try:
            yield target_db
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
    
    @pytest.fixture
    def phenology_system(self, phenology_db_root):
        """Set up complete phenology system."""
        manager = UnifiedParameterManager(database_path=phenology_db_root)
        manager.database.load_all()
        context = {
            'param_manager': manager,
        }
        yield context

    def test_alternative_phenology_closures(self, phenology_system):
        """Test each phenology closure in an independent graph selection."""
        with patch('relic.parameters.manager.get_global_manager', 
                   return_value=phenology_system['param_manager']):
            
            # Create phenology functors
            static_functor = StaticPhenologyFunctor(name="static")
            gdd_functor = GDDPhenologyFunctor(name="gdd")
            gsi_functor = GSIPhenologyFunctor(name="gsi")
            crop_functor = CropPhenologyFunctor(name="crop_phenology")
            
            # Each model is an alternative closure and uses its own graph.
            for functor in (
                static_functor,
                gdd_functor,
                gsi_functor,
                crop_functor,
            ):
                assert functor.get_dependencies() == []
                resolver = DependencyGraphResolver()
                resolver.add_functor(functor)
                assert set(resolver.nodes) == {functor.name}

            state = _canonical_plant_state()
            environment = {
                'air_temperature': torch.tensor(293.15),
                'day_of_year': torch.tensor(150.0),
                'air_temperature_min': torch.tensor(288.15),
                'air_temperature_max': torch.tensor(298.15),
                'vapor_pressure_deficit': torch.tensor(1500.0),
                'daylength': torch.tensor(50400.0),
            }
            
            # Test static phenology
            static_updates = static_functor(
                {**state, **environment},
                time=0.0,
                dt=SECONDS_PER_DAY,
            )
            static_result = {
                key: update.value for key, update in static_updates.items()
            }
            assert 'phenological_factor' in static_result
            static_factor = static_result['phenological_factor']
            assert torch.all(static_factor >= 0.0)
            assert torch.all(static_factor <= 1.0)
            
            # Test GDD phenology
            gdd_updates = gdd_functor(
                {
                    **state,
                    **environment,
                    'phenology_gdd_accumulated': torch.tensor(0.0),
                    'phenology_gdd_previous_day_of_year': torch.tensor(149.0),
                },
                time=0.0,
                dt=SECONDS_PER_DAY,
            )
            gdd_result = {
                key: update.value for key, update in gdd_updates.items()
            }
            assert 'phenological_factor' in gdd_result
            assert 'diagnostics.phenology.accumulated_gdd' in gdd_result
            assert 'thermal_time' not in gdd_result
            for state_name, expected_unit in (
                ('phenology_gdd_accumulated', 'degC day'),
                ('phenology_gdd_previous_day_of_year', 'day'),
            ):
                update = gdd_updates[state_name]
                assert isinstance(update, UpdateItem)
                assert update.mode is UpdateMode.REPLACE
                assert update.per_timestep is True
                assert update.unit == expected_unit
                assert update.source_functor == 'gdd'

            # Test GSI phenology
            gsi_updates = gsi_functor(
                {
                    **state,
                    **environment,
                    **_canonical_gsi_state(gsi_functor),
                },
                time=0.0,
                dt=SECONDS_PER_DAY,
            )
            gsi_result = {
                key: update.value for key, update in gsi_updates.items()
            }
            assert 'phenological_factor' in gsi_result
            assert 'diagnostics.phenology.growing_season_index' in gsi_result
            assert 'growing_season_index' not in gsi_result
            for state_name in _canonical_gsi_state(gsi_functor):
                assert gsi_updates[state_name].mode is UpdateMode.REPLACE
                assert gsi_updates[state_name].per_timestep is True

            # Test Crop phenology with complete external temporal state.
            crop_updates = crop_functor(
                {
                    **state,
                    **environment,
                    **_canonical_crop_state(),
                },
                time=0.0,
                dt=SECONDS_PER_DAY,
            )
            crop_units = {
                'phenology_crop_accumulated_phu': 'degC day',
                'phenology_crop_is_sown': 'dimensionless',
                'phenology_crop_is_harvested': 'dimensionless',
                'phenology_crop_vernalization_days': 'day',
            }
            for state_name, expected_unit in crop_units.items():
                update = crop_updates[state_name]
                assert update.mode is UpdateMode.REPLACE
                assert update.per_timestep is True
                assert update.unit == expected_unit
                assert update.source_functor == 'crop_phenology'
            assert crop_updates[
                'phenology_crop_accumulated_phu'
            ].value.item() == pytest.approx(15.0)
            assert 'phu_accumulated' not in crop_updates
            assert not hasattr(crop_functor, '_compute_allocation_fractions')
            
    def test_phenology_orchestrator_integration(self, phenology_system):
        """Verify the removed parallel orchestrator is not reintroduced."""
        assert PhenologyOrchestrator is None

    def test_phenology_functor_aggregates_canonical_owner_output(
        self, phenology_system
    ):
        """The system shell observes one real owner without recalculating it."""
        with patch(
            'relic.parameters.manager.get_global_manager',
            return_value=phenology_system['param_manager'],
        ):
            owner = GSIPhenologyFunctor(name="gsi_phenology")
            state = {
                **_canonical_plant_state(),
                'day_of_year': torch.tensor(150.0),
                'air_temperature': torch.tensor(293.15),
            }
            environment = {
                'day_of_year': torch.tensor(150.0),
                'air_temperature': torch.tensor(293.15),
                'air_temperature_min': torch.tensor(288.15),
                'vapor_pressure_deficit': torch.tensor(1500.0),
                'daylength': torch.tensor(50400.0),
                **_canonical_gsi_state(owner),
            }
            owner_updates = owner(
                {**state, **environment},
                time=0.0,
                dt=SECONDS_PER_DAY,
            )
            owner_output = {
                key: update.value for key, update in owner_updates.items()
            }
            functor = PhenologyFunctor(
                config={"canonical_model_owner": "gsi_phenology"}
            )

            observations = functor.forward(
                state,
                time=0.0,
                dt=SECONDS_PER_DAY,
                subprocess_outputs={"gsi_phenology": owner_output},
            )

        expected_observations = {
            'diagnostics.phenology_state': (
                'phenological_factor',
                'dimensionless',
            ),
            'diagnostics.growth_stage': (
                'phenology_phase',
                'dimensionless',
            ),
            'diagnostics.leaf_area_index': (
                'leaf_area_index',
                'm2 m-2',
            ),
            'diagnostics.phenology.senescence_factor': (
                'phenology_senescence_factor',
                'dimensionless',
            ),
        }
        for diagnostic, (source, unit) in expected_observations.items():
            item = observations[diagnostic]
            assert isinstance(item, UpdateItem)
            assert item.value is owner_output[source]
            assert item.mode is UpdateMode.REPLACE
            assert item.per_timestep is True
            assert item.unit == unit
            assert item.source_functor == functor.name
        assert 'diagnostics.senescence_rate' not in observations
        assert set(observations) == set(functor.metadata.output_variables)
            
    def test_parameter_changes_propagation(self, phenology_system):
        """Test that parameter changes propagate to phenology modules."""
        with patch('relic.parameters.manager.get_global_manager',
                   return_value=phenology_system['param_manager']):
            
            # Create functor
            functor = StaticPhenologyFunctor(name="test_propagation")
            
            # Get initial parameter value
            initial_steepness = functor.get_parameter_value(
                'static.transition_steepness',
                namespace='biosphere.phenology'
            )
            assert initial_steepness is not None
            if isinstance(initial_steepness, torch.Tensor):
                initial_value = float(initial_steepness.detach().cpu().item())
            else:
                initial_value = float(initial_steepness)
            new_value = initial_value + 0.05
            
            # Update parameter
            canonical_spec = phenology_system['param_manager'].get_parameter(
                'biosphere.phenology.static.transition_steepness'
            )
            spec = replace(canonical_spec, value=new_value)
            phenology_system['param_manager'].set_parameter(spec, tier="calibrated")
            
            # Get updated value
            updated_steepness = functor.get_parameter_value(
                'static.transition_steepness',
                namespace='biosphere.phenology'
            )
            if isinstance(updated_steepness, torch.Tensor):
                updated_value = float(updated_steepness.detach().cpu().item())
            else:
                updated_value = float(updated_steepness)
            assert updated_value == pytest.approx(new_value)
            
    def test_multi_species_phenology(self, phenology_system):
        """Test phenology for multiple species/PFTs."""
        with patch('relic.parameters.manager.get_global_manager',
                   return_value=phenology_system['param_manager']):
            
            species_configs = [
                {'name': 'deciduous_tree', 'max_lai': 7.0},
                {'name': 'evergreen_tree', 'max_lai': 5.0},
                {'name': 'grass_c3', 'max_lai': 3.0}
            ]
            
            results = {}
            
            for config in species_configs:
                # Override max_lai for each species
                canonical_spec = phenology_system['param_manager'].get_parameter(
                    'biosphere.phenology.morphology.max_lai'
                )
                spec = replace(canonical_spec, value=config['max_lai'])
                phenology_system['param_manager'].set_parameter(spec, tier="calibrated")
                
                # Create functor for species
                functor = StaticPhenologyFunctor(name=config['name'])
                
                state = _canonical_plant_state()
                
                environment = {
                    'air_temperature': torch.tensor(293.15),
                    'day_of_year': torch.tensor(150.0),
                }
                
                updates = functor(
                    {**state, **environment},
                    time=0.0,
                    dt=86400.0,
                )
                result = {key: update.value for key, update in updates.items()}
                results[config['name']] = result
                
            # Verify all species computed
            assert len(results) == 3
            for species_name in results:
                assert 'phenological_factor' in results[species_name]
                
    def test_missing_required_state_fails_fast(self, phenology_system):
        """Test that missing vegetation pools cannot trigger hidden defaults."""
        with patch(
            'relic.parameters.manager.get_global_manager',
            return_value=phenology_system['param_manager'],
        ):
            functor = StaticPhenologyFunctor(name="error_test")
            state = {
                'day_of_year': torch.tensor(150.0),
                'air_temperature': torch.tensor(293.15),
            }
            with pytest.raises(
                ExecutionError,
                match="Required state variable 'leaf_carbon' not found",
            ):
                functor(state, time=0.0, dt=86400.0)
            
    def test_parameter_audit_complete_system(self, phenology_system):
        """Test parameter audit for complete phenology system."""
        with patch('relic.parameters.manager.get_global_manager',
                   return_value=phenology_system['param_manager']):
            
            # Create various phenology components
            static = StaticPhenologyFunctor(name="static")
            gdd = GDDPhenologyFunctor(name="gdd")
            gsi = GSIPhenologyFunctor(name="gsi")
            
            # Access parameters to generate requests
            static.get_parameter_value('static.transition_steepness')
            gdd.get_parameter_value('gdd.base_temperature')
            gsi.get_parameter_value('gsi.memory_days')
            static.get_parameter_value('static.sigmoid_steepness')
            static.get_parameter_value('static.sigmoid_midpoint')
            gdd.get_parameter_value('gdd.threshold_start')
            gdd.get_parameter_value('gdd.threshold_end')
            gsi.get_parameter_value('gsi.tmin_threshold')
            gsi.get_parameter_value('gsi.vpd_min')
            
            # Missing required parameters must not use a hidden fallback.
            with pytest.raises((KeyError, ValueError)):
                static.get_parameter_value('nonexistent.param', required=True)
            
            # Run audit
            audit_result = phenology_system['param_manager'].run_registry_audit()
            
            # Check audit completeness
            assert audit_result.observed_total > 20  # Should have many parameters
            
            # Check for any missing parameters
            # Export audit report
            report_dict = audit_result.to_dict()
            assert 'summary' in report_dict
            assert 'issues' in report_dict
            
    def test_conservation_with_parameters(self, phenology_system):
        """Test conservation laws with parameterized phenology."""
        with patch('relic.parameters.manager.get_global_manager',
                   return_value=phenology_system['param_manager']):
            
            functor = StaticPhenologyFunctor(name="conservation_test")
            
            state = _canonical_plant_state()
            
            environment = {
                'air_temperature': torch.tensor(293.15),
                'day_of_year': torch.tensor(280.0),
            }

            dt = 3600.0
            result = functor(
                {**state, **environment},
                time=0.0,
                dt=dt,
            )
            for key in (
                'leaf_carbon',
                'froot_carbon',
                'leaf_nitrogen',
                'froot_nitrogen',
                'plant_nitrogen_pool',
            ):
                assert isinstance(result[key], UpdateItem)
                assert result[key].mode is UpdateMode.ACCUMULATE
                assert result[key].per_timestep is True

            carbon_residual = result[
                'diagnostics.phenology.carbon_turnover_balance_residual'
            ].value
            nitrogen_residual = result[
                'diagnostics.phenology.nitrogen_turnover_balance_residual'
            ].value
            assert torch.allclose(carbon_residual, torch.zeros_like(carbon_residual))
            assert torch.allclose(
                nitrogen_residual,
                torch.zeros_like(nitrogen_residual),
                atol=1e-10,
                rtol=0.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



