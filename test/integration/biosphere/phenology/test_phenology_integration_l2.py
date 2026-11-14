"""
L2 Integration Tests for Phenology Module

Tests integration with StateManager, conservation laws, and coupling
with other biosphere modules.

Author: Yuchong Long  
Date: 2025-01-24
"""

import pytest
import torch
import warnings

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from relic.scientific.biosphere.phenology.models import (
    StaticPhenology, GDDPhenology, GSIPhenology, CropPhenology
)
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant


def _gsi_temporal_state(model, *, device=None):
    """Return canonical StateManager-owned GSI temporal inputs."""
    return {
        'phenology_gsi_history': torch.zeros(model.memory_days, device=device),
        'phenology_gsi_history_index': torch.tensor(0.0, device=device),
        'phenology_gsi_history_count': torch.tensor(0.0, device=device),
        'phenology_gsi_in_growing_season': torch.tensor(0.0, device=device),
    }


CROP_TEMPORAL_STATE_UNITS = {
    'phenology_crop_accumulated_phu': 'degC day',
    'phenology_crop_is_sown': 'dimensionless',
    'phenology_crop_is_harvested': 'dimensionless',
    'phenology_crop_vernalization_days': 'day',
}


def _crop_temporal_state(*, device=None):
    """Return canonical StateManager-owned Crop temporal inputs."""
    return {
        'phenology_crop_accumulated_phu': torch.tensor(0.0, device=device),
        'phenology_crop_is_sown': torch.tensor(0.0, device=device),
        'phenology_crop_is_harvested': torch.tensor(0.0, device=device),
        'phenology_crop_vernalization_days': torch.tensor(0.0, device=device),
    }


class TestPhenologyStateIntegration:
    """Test phenology integration with StateManager."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database('relic/biosphere')
        

    def test_state_update_conservation(self):
        """Test integrated carbon updates against explicit boundary exports."""
        config = dict({'phenology_type': 'static'})
        phenology = StaticPhenology('test_conservation', config)
        initial_state = {
            'day_of_year': torch.tensor(280.0),
            'air_temperature': torch.tensor(283.15),
            'leaf_carbon': torch.tensor(2.0),
            'froot_carbon': torch.tensor(1.5),
            'leaf_nitrogen': torch.tensor(0.08),
            'froot_nitrogen': torch.tensor(0.04),
            'plant_nitrogen_pool': torch.tensor(0.2),
        }

        dt = 1800.0  # 30 minutes in seconds
        updates = phenology(
            initial_state,
            time=0.0,
            dt=dt,
        )

        for key in ('leaf_carbon', 'froot_carbon'):
            assert isinstance(updates[key], UpdateItem)
            assert updates[key].mode is UpdateMode.ACCUMULATE
            assert updates[key].per_timestep is True
            assert updates[key].element == 'C'

        for key in ('leaf_nitrogen', 'froot_nitrogen', 'plant_nitrogen_pool'):
            assert isinstance(updates[key], UpdateItem)
            assert updates[key].mode is UpdateMode.ACCUMULATE
            assert updates[key].per_timestep is True
            assert updates[key].element == 'N'

        for key in ('leaf_litterfall_carbon', 'froot_litterfall_carbon'):
            assert updates[key].element == 'C'

        for key in (
            'leaf_litterfall_nitrogen',
            'froot_litterfall_nitrogen',
        ):
            assert updates[key].element == 'N'
        assert updates['nitrogen_retranslocation'].element is None

        carbon_residual = (
            updates['leaf_carbon'].value
            + updates['froot_carbon'].value
            + dt
            * (
                updates['leaf_litterfall_carbon'].value
                + updates['froot_litterfall_carbon'].value
            )
        )
        expected_residual = updates[
            'diagnostics.phenology.carbon_turnover_balance_residual'
        ].value
        assert torch.allclose(carbon_residual, torch.zeros_like(carbon_residual))
        assert torch.allclose(carbon_residual, expected_residual)
        assert updates['leaf_carbon'].value <= 0
        assert updates['froot_carbon'].value <= 0
        
    def test_multi_model_consistency(self):
        """Test consistency across different phenology models."""
        gsi_model = GSIPhenology('gsi', dict({'phenology_type': 'gsi'}))
        models = [
            StaticPhenology('static', dict({'phenology_type': 'static'})),
            GDDPhenology('gdd', dict({'phenology_type': 'gdd'})),
            gsi_model,
        ]
        
        # Common state and environment
        state = {
            'leaf_carbon': torch.tensor(1.0),
            'froot_carbon': torch.tensor(1.0),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02)
        }
        
        environment = {
            'day_of_year': torch.tensor(200.0),
            'air_temperature': torch.tensor(293.15),
            'air_temperature_min': torch.tensor(288.15),
            'air_temperature_max': torch.tensor(298.15),
            'vapor_pressure_deficit': torch.tensor(1500.0),
            'daylength': torch.tensor(50000.0),
            'phenology_gdd_accumulated': torch.tensor(0.0),
            'phenology_gdd_previous_day_of_year': torch.tensor(199.0),
            **_gsi_temporal_state(gsi_model),
        }
        seconds_per_day = get_physical_constant("seconds_per_day")

        for model in models:
            updates = model(
                {**state, **environment},
                time=0.0,
                dt=seconds_per_day,
            )
            fluxes = {key: update.value for key, update in updates.items()}
            
            # All models should produce these keys
            assert 'leaf_area_index' in fluxes
            assert 'phenological_factor' in fluxes
            assert 'phenology_senescence_factor' in fluxes
            
            # All values should be valid
            for key, value in fluxes.items():
                if isinstance(value, torch.Tensor):
                    assert not torch.isnan(value).any(), f"NaN in {key} for {model.__class__.__name__}"
                    assert not torch.isinf(value).any(), f"Inf in {key} for {model.__class__.__name__}"
                    
            # LAI should be non-negative
            assert fluxes['leaf_area_index'] >= 0
            
            # Factors should be in [0, 1]
            assert 0 <= fluxes['phenological_factor'] <= 1
            assert 0 <= fluxes['phenology_senescence_factor'] <= 1
            

class TestPhenologyConservation:
    """Test conservation laws in phenology processes."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database('relic/biosphere')
        
    def test_carbon_flux_conservation(self):
        """Test that carbon fluxes are balanced."""
        config = dict({'phenology_type': 'gdd'})
        phenology = GDDPhenology('test_c_conservation', config)
        
        state = {
            'leaf_carbon': torch.tensor(3.0),
            'froot_carbon': torch.tensor(2.0),
            'leaf_nitrogen': torch.tensor(0.12),
            'froot_nitrogen': torch.tensor(0.06)
        }
        
        environment = {
            'day_of_year': torch.tensor(270.0),
            'air_temperature': torch.tensor(278.15),  # 5 degC - cold
            'air_temperature_min': torch.tensor(273.15),
            'air_temperature_max': torch.tensor(283.15),
            'phenology_gdd_accumulated': torch.tensor(1000.0),
            'phenology_gdd_previous_day_of_year': torch.tensor(269.0),
        }
        
        updates = phenology(
            {**state, **environment},
            time=0.0,
            dt=1.0,
        )
        fluxes = {key: update.value for key, update in updates.items()}
        
        # Sum carbon fluxes
        carbon_loss = torch.tensor(0.0)
        carbon_gain = torch.tensor(0.0)
        
        # Plant pool losses (negative increments)
        if 'leaf_carbon' in fluxes:
            carbon_loss += -fluxes['leaf_carbon']  # Negative increment = loss
        if 'froot_carbon' in fluxes:
            carbon_loss += -fluxes['froot_carbon']
            
        # Litter pool gains (positive increments)  
        if 'leaf_litter_carbon' in fluxes:
            carbon_gain += fluxes['leaf_litter_carbon']
        if 'froot_litter_carbon' in fluxes:
            carbon_gain += fluxes['froot_litter_carbon']
            
        # Conservation check
        if carbon_loss > 0 and carbon_gain > 0:
            balance = abs(carbon_loss - carbon_gain) / carbon_loss
            assert balance < 0.01, f"Carbon imbalance: loss={carbon_loss}, gain={carbon_gain}, error={balance*100}%"
            
    def test_nitrogen_flux_conservation(self):
        """Test integrated nitrogen updates with retranslocation and litter export."""
        config = dict({'phenology_type': 'static'})
        phenology = StaticPhenology('test_n_conservation', config)
        
        state = {
            'day_of_year': torch.tensor(290.0),
            'air_temperature': torch.tensor(278.15),
            'leaf_carbon': torch.tensor(2.0),
            'froot_carbon': torch.tensor(1.5),
            'leaf_nitrogen': torch.tensor(0.1),  # High N content
            'froot_nitrogen': torch.tensor(0.05),
            'plant_nitrogen_pool': torch.tensor(0.2)
        }

        dt = 1800.0
        updates = phenology(
            state,
            time=0.0,
            dt=dt,
        )

        nitrogen_residual = (
            updates['leaf_nitrogen'].value
            + updates['froot_nitrogen'].value
            + updates['plant_nitrogen_pool'].value
            + dt
            * (
                updates['leaf_litterfall_nitrogen'].value
                + updates['froot_litterfall_nitrogen'].value
            )
        )
        expected_residual = updates[
            'diagnostics.phenology.nitrogen_turnover_balance_residual'
        ].value
        assert torch.allclose(
            nitrogen_residual,
            torch.zeros_like(nitrogen_residual),
            atol=1e-10,
            rtol=0.0,
        )
        assert torch.allclose(nitrogen_residual, expected_residual)
        assert updates['plant_nitrogen_pool'].mode is UpdateMode.ACCUMULATE
        assert updates['plant_nitrogen_pool'].per_timestep is True
                

class TestPhenologyTemporalContinuity:
    """Test temporal continuity and stability of phenology models."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database('relic/biosphere')
        
    def test_gdd_accumulation_continuity(self):
        """Checkpointed GDD state must reproduce the uninterrupted trajectory."""
        phenology = GDDPhenology()
        state_manager = StateManager(device="cpu")
        state_names = (
            'phenology_gdd_accumulated',
            'phenology_gdd_previous_day_of_year',
        )
        expected_contracts = {
            'phenology_gdd_accumulated': 'degC day',
            'phenology_gdd_previous_day_of_year': 'day',
        }
        registry = state_manager._contract_registry
        assert registry is not None
        for state_name, unit in expected_contracts.items():
            contract = registry.get_contract(state_name)
            assert contract is not None
            assert contract.kind == 'state'
            assert contract.udunits == unit
            assert contract.owner == 'gdd_phenology'
            assert contract.writers == ['gdd_phenology']
            assert contract.incrementable is False
            assert state_manager._schema[state_name].incrementable is False

        state_manager.initialize_state({
            'phenology_gdd_accumulated': torch.tensor(0.0),
            'phenology_gdd_previous_day_of_year': torch.tensor(99.0),
        })
        seconds_per_day = float(get_physical_constant("seconds_per_day"))
        plant_state = {
            'leaf_carbon': torch.tensor(1.0),
            'froot_carbon': torch.tensor(1.0),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02),
        }

        def advance(
            manager: StateManager,
            functor: GDDPhenology,
            start_day: int,
            stop_day: int,
        ) -> dict[str, UpdateItem]:
            last_updates = None
            for day in range(start_day, stop_day):
                current = manager.get_current_state()
                prior_gdd = current['phenology_gdd_accumulated'].clone()
                updates = functor(
                    {
                        **plant_state,
                        **current,
                        'day_of_year': torch.tensor(float(day)),
                        'air_temperature': torch.tensor(288.15),
                        'air_temperature_min': torch.tensor(283.15),
                        'air_temperature_max': torch.tensor(293.15),
                    },
                    time=(day - 1) * seconds_per_day,
                    dt=seconds_per_day,
                )
                temporal_updates = {
                    name: updates[name] for name in state_names
                }
                manager.apply_updates(
                    temporal_updates,
                    dt=seconds_per_day,
                    check_conservation=False,
                    update_sources={
                        name: 'gdd_phenology' for name in state_names
                    },
                )
                assert manager.get_state(
                    'phenology_gdd_accumulated'
                ).item() >= prior_gdd.item()
                last_updates = updates
            assert last_updates is not None
            return last_updates

        advance(state_manager, phenology, 100, 150)
        checkpoint = state_manager.checkpoint()
        checkpoint_gdd = state_manager.get_state(
            'phenology_gdd_accumulated'
        ).clone()
        assert checkpoint_gdd.item() == pytest.approx(500.0)

        uninterrupted_updates = advance(state_manager, phenology, 150, 200)
        uninterrupted = {
            name: state_manager.get_state(name).clone() for name in state_names
        }

        restarted_manager = StateManager(device="cpu")
        restarted_manager.restore(checkpoint)
        restarted_phenology = GDDPhenology()
        resumed_updates = advance(
            restarted_manager,
            restarted_phenology,
            150,
            200,
        )
        resumed = {
            name: restarted_manager.get_state(name).clone() for name in state_names
        }

        assert uninterrupted['phenology_gdd_accumulated'].item() == pytest.approx(1000.0)
        assert uninterrupted['phenology_gdd_previous_day_of_year'].item() == pytest.approx(199.0)
        for state_name in state_names:
            torch.testing.assert_close(resumed[state_name], uninterrupted[state_name])
        for output_name in (
            'leaf_area_index',
            'phenological_factor',
            'phenology_phase',
        ):
            torch.testing.assert_close(
                resumed_updates[output_name].value,
                uninterrupted_updates[output_name].value,
            )

    def test_real_simulator_commits_gdd_state_and_provenance(self):
        """Commit canonical GDD full states through the real simulator."""
        dt = float(get_physical_constant("seconds_per_day"))
        t_freeze = float(get_physical_constant("T_FREEZE"))
        initial_state = {
            'day_of_year': torch.tensor(150.0),
            'air_temperature': torch.tensor(t_freeze + 15.0),
            'air_temperature_max': torch.tensor(t_freeze + 20.0),
            'air_temperature_min': torch.tensor(t_freeze + 10.0),
            'phenology_gdd_accumulated': torch.tensor(42.0),
            'phenology_gdd_previous_day_of_year': torch.tensor(149.0),
            'leaf_carbon': torch.tensor(1.0),
            'froot_carbon': torch.tensor(1.0),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02),
            'plant_nitrogen_pool': torch.tensor(0.0),
        }
        direct = GDDPhenology().forward(initial_state, time=0.0, dt=dt)
        simulator = RELICSimulator({
            'time_step': dt,
            'max_time': dt,
            'checkpoint_frequency': 2.0 * dt,
            'device': 'cpu',
            'enable_auto_scale': False,
            'conservation_checks': False,
            'strict_dependencies': True,
            'strict_scale_detection': False,
        })
        simulator.initialize_state({
            key: value.clone() for key, value in initial_state.items()
        })
        runtime_functor = GDDPhenology()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            simulator.register_functor(runtime_functor, execution_group='fast')
            updates = simulator.step(dt=dt)

        expected = {
            'phenology_gdd_accumulated': (52.0, 'degC day'),
            'phenology_gdd_previous_day_of_year': (150.0, 'day'),
        }
        for state_name, (expected_value, unit) in expected.items():
            direct_update = direct[state_name]
            assert direct_update.mode is UpdateMode.REPLACE
            assert direct_update.per_timestep is True
            assert direct_update.unit == unit
            assert direct_update.source_functor == 'gdd_phenology'
            assert direct_update.value.item() == pytest.approx(expected_value)
            runtime_value = updates[state_name]
            expected_runtime_value = direct_update.value.to(
                device=runtime_value.device,
                dtype=runtime_value.dtype,
            )
            torch.testing.assert_close(runtime_value, expected_runtime_value)
            torch.testing.assert_close(
                simulator.state_manager.get_state(state_name),
                expected_runtime_value,
            )
            spec = simulator.state_manager.get_variable_spec(state_name)
            assert spec is not None
            assert spec.units == unit
            assert spec.incrementable is False
            provenance = simulator.get_variable_provenance(state_name)
            assert {entry['functor'] for entry in provenance} == {'gdd_phenology'}
            assert {entry['role'] for entry in provenance} == {
                'writer',
                'final_writer',
            }
            assert {entry['mode'] for entry in provenance} == {
                'UpdateMode.REPLACE'
            }

        diagnostic_name = 'diagnostics.phenology.accumulated_gdd'
        assert diagnostic_name in updates
        assert diagnostic_name not in simulator.state_manager.get_current_state()
        assert simulator.get_variable_provenance(diagnostic_name) == []
        assert simulator.get_effective_device('gdd_phenology').type == 'cpu'
        assert caught == []

    def test_gsi_checkpoint_restart_reproduces_temporal_state(self):
        """Checkpointed GSI state must reproduce an uninterrupted trajectory."""
        phenology = GSIPhenology()
        state_manager = StateManager(device="cpu")
        state_names = (
            'phenology_gsi_history',
            'phenology_gsi_history_index',
            'phenology_gsi_history_count',
            'phenology_gsi_in_growing_season',
        )
        registry = state_manager._contract_registry
        assert registry is not None
        for state_name in state_names:
            contract = registry.get_contract(state_name)
            assert contract is not None
            assert contract.kind == 'state'
            assert contract.udunits == 'dimensionless'
            assert contract.owner == 'gsi_phenology'
            assert contract.writers == ['gsi_phenology']
            assert contract.incrementable is False
            assert state_manager._schema[state_name].incrementable is False

        state_manager.initialize_state(_gsi_temporal_state(phenology))
        seconds_per_day = float(get_physical_constant("seconds_per_day"))
        freeze = float(get_physical_constant("T_FREEZE"))
        plant_state = {
            'leaf_carbon': torch.tensor(1.0),
            'froot_carbon': torch.tensor(1.0),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02),
        }

        def advance(
            manager: StateManager,
            functor: GSIPhenology,
            start_day: int,
            stop_day: int,
        ) -> dict[str, UpdateItem]:
            last_updates = None
            for day in range(start_day, stop_day):
                current = manager.get_current_state()
                updates = functor(
                    {
                        **plant_state,
                        **current,
                        'day_of_year': torch.tensor(float(day)),
                        'air_temperature': torch.tensor(freeze + 15.0),
                        'air_temperature_min': torch.tensor(freeze + 5.0),
                        'vapor_pressure_deficit': torch.tensor(
                            900.0 + 300.0 * float(day % 5)
                        ),
                        'daylength': torch.tensor(50000.0),
                    },
                    time=(day - 1) * seconds_per_day,
                    dt=seconds_per_day,
                )
                manager.apply_updates(
                    {name: updates[name] for name in state_names},
                    dt=seconds_per_day,
                    check_conservation=False,
                    update_sources={name: 'gsi_phenology' for name in state_names},
                )
                last_updates = updates
            assert last_updates is not None
            return last_updates

        advance(state_manager, phenology, 100, 104)
        checkpoint = state_manager.checkpoint()
        uninterrupted_updates = advance(state_manager, phenology, 104, 112)
        uninterrupted = {
            name: state_manager.get_state(name).clone() for name in state_names
        }

        restarted_manager = StateManager(device="cpu")
        restarted_manager.restore(checkpoint)
        restarted_phenology = GSIPhenology()
        resumed_updates = advance(
            restarted_manager,
            restarted_phenology,
            104,
            112,
        )
        resumed = {
            name: restarted_manager.get_state(name).clone() for name in state_names
        }

        for state_name in state_names:
            torch.testing.assert_close(resumed[state_name], uninterrupted[state_name])
        for output_name in (
            'leaf_area_index',
            'phenological_factor',
            'phenology_phase',
            'diagnostics.phenology.growing_season_index',
        ):
            torch.testing.assert_close(
                resumed_updates[output_name].value,
                uninterrupted_updates[output_name].value,
            )

    def test_real_simulator_commits_gsi_state_and_provenance(self):
        """Commit canonical GSI full states through the real simulator."""
        dt = float(get_physical_constant("seconds_per_day"))
        freeze = float(get_physical_constant("T_FREEZE"))
        direct_functor = GSIPhenology()
        initial_state = {
            'day_of_year': torch.tensor(150.0),
            'air_temperature': torch.tensor(freeze + 15.0),
            'air_temperature_min': torch.tensor(freeze + 5.0),
            'vapor_pressure_deficit': torch.tensor(900.0),
            'daylength': torch.tensor(50000.0),
            **_gsi_temporal_state(direct_functor),
            'leaf_carbon': torch.tensor(1.0),
            'froot_carbon': torch.tensor(1.0),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02),
            'plant_nitrogen_pool': torch.tensor(0.0),
        }
        direct = direct_functor.forward(initial_state, time=0.0, dt=dt)
        simulator = RELICSimulator({
            'time_step': dt,
            'max_time': dt,
            'checkpoint_frequency': 2.0 * dt,
            'device': 'cpu',
            'enable_auto_scale': False,
            'conservation_checks': False,
            'strict_dependencies': True,
            'strict_scale_detection': False,
        })
        simulator.initialize_state({
            key: value.clone() for key, value in initial_state.items()
        })
        daylength_spec = simulator.state_manager.get_variable_spec('daylength')
        assert daylength_spec is not None
        assert daylength_spec.units == 's'
        assert daylength_spec.bounds == (0.0, 86400.0)
        runtime_functor = GSIPhenology()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            simulator.register_functor(runtime_functor, execution_group='fast')
            updates = simulator.step(dt=dt)

        state_names = (
            'phenology_gsi_history',
            'phenology_gsi_history_index',
            'phenology_gsi_history_count',
            'phenology_gsi_in_growing_season',
        )
        for state_name in state_names:
            direct_update = direct[state_name]
            assert direct_update.mode is UpdateMode.REPLACE
            assert direct_update.per_timestep is True
            assert direct_update.unit == 'dimensionless'
            assert direct_update.source_functor == 'gsi_phenology'
            runtime_value = updates[state_name]
            expected_value = direct_update.value.to(
                device=runtime_value.device,
                dtype=runtime_value.dtype,
            )
            torch.testing.assert_close(runtime_value, expected_value)
            torch.testing.assert_close(
                simulator.state_manager.get_state(state_name),
                expected_value,
            )
            spec = simulator.state_manager.get_variable_spec(state_name)
            assert spec is not None
            assert spec.units == 'dimensionless'
            assert spec.incrementable is False
            provenance = simulator.get_variable_provenance(state_name)
            assert {entry['functor'] for entry in provenance} == {
                'gsi_phenology'
            }
            assert {entry['role'] for entry in provenance} == {
                'writer',
                'final_writer',
            }
            assert {entry['mode'] for entry in provenance} == {
                'UpdateMode.REPLACE'
            }

        diagnostic_name = 'diagnostics.phenology.growing_season_index'
        assert diagnostic_name in updates
        assert diagnostic_name not in simulator.state_manager.get_current_state()
        assert simulator.get_variable_provenance(diagnostic_name) == []
        assert simulator.get_effective_device('gsi_phenology').type == 'cpu'
        assert caught == []

    def test_crop_checkpoint_restart_reproduces_temporal_state(self):
        """Checkpointed Crop state reproduces an uninterrupted trajectory."""
        phenology = CropPhenology()
        state_manager = StateManager(device='cpu')
        state_names = tuple(CROP_TEMPORAL_STATE_UNITS)
        registry = state_manager._contract_registry
        assert registry is not None
        for state_name, unit in CROP_TEMPORAL_STATE_UNITS.items():
            contract = registry.get_contract(state_name)
            assert contract is not None
            assert contract.kind == 'state'
            assert contract.udunits == unit
            assert contract.owner == 'crop_phenology'
            assert contract.writers == ['crop_phenology']
            assert contract.incrementable is False
            assert state_manager._schema[state_name].incrementable is False

        state_manager.initialize_state(_crop_temporal_state())
        seconds_per_day = float(get_physical_constant('seconds_per_day'))
        freeze = float(get_physical_constant('T_FREEZE'))
        plant_state = {
            'leaf_carbon': torch.tensor(1.0),
            'froot_carbon': torch.tensor(1.0),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02),
        }

        def advance(
            manager: StateManager,
            functor: CropPhenology,
            start_day: int,
            stop_day: int,
        ) -> dict[str, UpdateItem]:
            last_updates = None
            for day in range(start_day, stop_day):
                current = manager.get_current_state()
                updates = functor(
                    {
                        **plant_state,
                        **current,
                        'day_of_year': torch.tensor(float(day)),
                        'air_temperature': torch.tensor(freeze + 20.0),
                    },
                    time=(day - 1) * seconds_per_day,
                    dt=seconds_per_day,
                )
                manager.apply_updates(
                    {name: updates[name] for name in state_names},
                    dt=seconds_per_day,
                    check_conservation=False,
                    update_sources={
                        name: 'crop_phenology' for name in state_names
                    },
                )
                last_updates = updates
            assert last_updates is not None
            return last_updates

        advance(state_manager, phenology, 118, 124)
        checkpoint = state_manager.checkpoint()
        checkpoint_phu = state_manager.get_state(
            'phenology_crop_accumulated_phu'
        ).clone()
        assert checkpoint_phu.item() == pytest.approx(60.0)
        uninterrupted_updates = advance(state_manager, phenology, 124, 130)
        uninterrupted = {
            name: state_manager.get_state(name).clone() for name in state_names
        }

        restarted_manager = StateManager(device='cpu')
        restarted_manager.restore(checkpoint)
        restarted_phenology = CropPhenology()
        resumed_updates = advance(
            restarted_manager,
            restarted_phenology,
            124,
            130,
        )
        resumed = {
            name: restarted_manager.get_state(name).clone()
            for name in state_names
        }

        for state_name in state_names:
            torch.testing.assert_close(
                resumed[state_name],
                uninterrupted[state_name],
            )
        for output_name in (
            'leaf_area_index',
            'phenological_factor',
            'phenology_phase',
            'harvest_index',
            'diagnostics.phenology.accumulated_phu',
        ):
            torch.testing.assert_close(
                resumed_updates[output_name].value,
                uninterrupted_updates[output_name].value,
            )
        assert not hasattr(restarted_phenology, 'accumulated_phu')
        assert not hasattr(restarted_phenology, 'is_sown')

    def test_real_simulator_commits_crop_state_and_provenance(self):
        """Commit canonical Crop full states through the real simulator."""
        dt = float(get_physical_constant('seconds_per_day'))
        freeze = float(get_physical_constant('T_FREEZE'))
        initial_state = {
            'day_of_year': torch.tensor(120.0),
            'air_temperature': torch.tensor(freeze + 20.0),
            **_crop_temporal_state(),
            'leaf_carbon': torch.tensor(1.0),
            'froot_carbon': torch.tensor(1.0),
            'leaf_nitrogen': torch.tensor(0.04),
            'froot_nitrogen': torch.tensor(0.02),
            'plant_nitrogen_pool': torch.tensor(0.0),
        }
        direct = CropPhenology().forward(initial_state, time=0.0, dt=dt)
        simulator = RELICSimulator({
            'time_step': dt,
            'max_time': dt,
            'checkpoint_frequency': 2.0 * dt,
            'device': 'cpu',
            'enable_auto_scale': False,
            'conservation_checks': False,
            'strict_dependencies': True,
            'strict_scale_detection': False,
        })
        simulator.initialize_state({
            key: value.clone() for key, value in initial_state.items()
        })
        runtime_functor = CropPhenology()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            simulator.register_functor(runtime_functor, execution_group='fast')
            updates = simulator.step(dt=dt)

        expected_values = {
            'phenology_crop_accumulated_phu': 15.0,
            'phenology_crop_is_sown': 1.0,
            'phenology_crop_is_harvested': 0.0,
            'phenology_crop_vernalization_days': 0.0,
        }
        for state_name, unit in CROP_TEMPORAL_STATE_UNITS.items():
            direct_update = direct[state_name]
            assert direct_update.mode is UpdateMode.REPLACE
            assert direct_update.per_timestep is True
            assert direct_update.unit == unit
            assert direct_update.source_functor == 'crop_phenology'
            assert direct_update.value.item() == pytest.approx(
                expected_values[state_name]
            )
            runtime_value = updates[state_name]
            expected_value = direct_update.value.to(
                device=runtime_value.device,
                dtype=runtime_value.dtype,
            )
            torch.testing.assert_close(runtime_value, expected_value)
            torch.testing.assert_close(
                simulator.state_manager.get_state(state_name),
                expected_value,
            )
            spec = simulator.state_manager.get_variable_spec(state_name)
            assert spec is not None
            assert spec.units == unit
            assert spec.incrementable is False
            provenance = simulator.get_variable_provenance(state_name)
            assert {entry['functor'] for entry in provenance} == {
                'crop_phenology'
            }
            assert {entry['role'] for entry in provenance} == {
                'writer',
                'final_writer',
            }
            assert {entry['mode'] for entry in provenance} == {
                'UpdateMode.REPLACE'
            }

        diagnostic_name = 'diagnostics.phenology.accumulated_phu'
        assert diagnostic_name in updates
        assert diagnostic_name not in simulator.state_manager.get_current_state()
        assert simulator.get_variable_provenance(diagnostic_name) == []
        assert simulator.get_effective_device('crop_phenology').type == 'cpu'
        assert caught == []

    def test_phenology_phase_transitions(self):
        """Test smooth transitions between phenological phases."""
        config = dict({'phenology_type': 'static'})
        phenology = StaticPhenology('test_transitions', config)
        
        # Test transition period
        transition_days = range(110, 130)  # Around onset
        growth_factors = []
        
        for day in transition_days:
            environment = {
                'day_of_year': torch.tensor(float(day)),
                'air_temperature': torch.tensor(288.15)
            }
            
            output = phenology.compute_phenological_state(
                environment,
                dt=get_physical_constant("seconds_per_day"),
            )
            
            growth_factors.append(output.growth_factor.item())
            
        # Check for smooth transition (no jumps > 0.2)
        for i in range(1, len(growth_factors)):
            change = abs(growth_factors[i] - growth_factors[i-1])
            assert change < 0.2, f"Abrupt transition: {growth_factors[i-1]} -> {growth_factors[i]}"
            
        # Should be generally increasing during spring onset
        assert growth_factors[-1] > growth_factors[0], \
            "Growth factor should increase during spring onset"
            

class TestPhenologyStressResponse:
    """Test phenology response to environmental stress."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database('relic/biosphere')
        
    def test_vpd_stress_response(self):
        """Test the canonical GSI response to high atmospheric VPD."""
        config = dict({'phenology_type': 'gsi'})
        phenology = GSIPhenology('test_vpd_stress', config)
        
        # Normal conditions
        normal_env = {
            'day_of_year': torch.tensor(200.0),
            'air_temperature': torch.tensor(293.15),
            'air_temperature_min': torch.tensor(288.15),
            'vapor_pressure_deficit': torch.tensor(1000.0),  # Low VPD
            'daylength': torch.tensor(50000.0),
            **_gsi_temporal_state(phenology),
        }
        
        normal_output = phenology.compute_phenological_state(
            normal_env,
            dt=torch.tensor(
                get_physical_constant("seconds_per_day")
            ),
        )
        
        # High atmospheric demand lowers the VPD response factor.
        stress_env = normal_env.copy()
        stress_env['vapor_pressure_deficit'] = torch.tensor(4000.0)

        stress_output = phenology.compute_phenological_state(
            stress_env,
            dt=get_physical_constant("seconds_per_day"),
        )

        # GSI should be lower under high VPD.
        assert stress_output.growing_season_index < normal_output.growing_season_index, \
            "GSI should decrease under high VPD"

        # Growth factor should be reduced
        assert stress_output.growth_factor < normal_output.growth_factor, \
            "Growth should be reduced under high VPD"
            
    def test_cold_stress_response(self):
        """Test phenology response to cold stress."""
        config = dict({'phenology_type': 'gsi'})
        phenology = GSIPhenology('test_cold', config)
        
        # Warm conditions
        warm_env = {
            'day_of_year': torch.tensor(180.0),
            'air_temperature': torch.tensor(293.15),  # 20 degC
            'air_temperature_min': torch.tensor(288.15),  # 15 degC
            'vapor_pressure_deficit': torch.tensor(1500.0),
            'daylength': torch.tensor(50000.0),
            **_gsi_temporal_state(phenology),
        }
        
        warm_output = phenology.compute_phenological_state(
            warm_env,
            dt=torch.tensor(
                get_physical_constant("seconds_per_day")
            ),
        )
        
        # Cold conditions
        cold_env = warm_env.copy()
        cold_env['air_temperature_min'] = torch.tensor(273.15)  # 0 degC
        cold_env['air_temperature'] = torch.tensor(278.15)  # 5 degC
        
        cold_output = phenology.compute_phenological_state(
            cold_env,
            dt=get_physical_constant("seconds_per_day"),
        )
        
        # GSI should be lower in cold
        assert cold_output.growing_season_index < warm_output.growing_season_index, \
            "GSI should decrease under cold stress"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
