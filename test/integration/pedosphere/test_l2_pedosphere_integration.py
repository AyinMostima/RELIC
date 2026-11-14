"""
L2 Integration Tests for Pedosphere Module

This module provides comprehensive L2 integration tests for the pedosphere
module, testing interactions between different components within the module
and their interfaces with external modules.

Test Categories:
- L2.1: Decomposition-Carbon Pool Integration
- L2.2: Parameter Management Integration  
- L2.3: Environmental Driver Integration
- L2.4: Mass Balance Integration
- L2.5: Multi-pool Dynamics Integration
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import warnings

from test.utils.forward_call import call_forward

# Import core modules under test
from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.parameters import ParameterSpec
from relic.scientific.anthroposphere.management.grazing import GrazingManagement
from relic.scientific.biosphere.phenology.models import StaticPhenology
from relic.scientific.pedosphere.decomposition import FirstOrderDecomposition
from relic.scientific.pedosphere.carbon_pools.pools_matrix import PoolsMatrix
from relic.scientific.pedosphere.interface_mapping import LitterfallInterfaceMapper
from relic.scientific.pedosphere.nutrient_cycling.mineralization import (
    MineralizationProcess,
)
from relic.scientific.pedosphere.nutrient_cycling.nitrogen import (
    NitrogenCycling,
)

from relic.utils.physical_constants import get_physical_constant


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


class TestPedosphereL2Integration:
    """L2 Integration Tests for Pedosphere Module Components."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary YAML configuration file."""
        config_content = """
pedosphere:
  decomposition:
    k_metabolic: 10.0  # per year
    k_structural: 4.9
    k_fast_som: 6.0
    k_slow_som: 0.2
    k_passive_som: 0.0045
    lignin_n_threshold: 0.45
    clay_content: 0.25
    temp_response_type: 'q10'
    moisture_response_power: 1.0
    field_capacity: 0.25
    wilting_point: 0.10

  physical:
    q10_decomposition: 2.0
    
  carbon_pools:
    pool_names: 
      - 'metabolic_litter'
      - 'structural_litter'
      - 'fast_som'
      - 'slow_som'
      - 'passive_som'
    mass_balance_tolerance: 1e-6
    min_pool_size: 1e-8
    max_pool_size: 1e3
    
  transfer_matrix:
    t_met_fast: 0.45
    t_met_slow: 0.45
    t_str_fast: 0.40
    t_str_slow: 0.40
    t_fast_slow: 0.42
    t_fast_passive: 0.03
    t_slow_fast: 0.45
    t_slow_passive: 0.03
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    @pytest.fixture
    def parameter_manager(self, temp_config_file):
        """Create UnifiedParameterManager with test configuration."""
        from relic.parameters import get_global_manager, reset_global_manager
        
        # Reset global manager to start fresh for tests
        reset_global_manager()
        manager = get_global_manager()
        
        # Set test parameters directly using the manager with proper namespacing
        test_params = {
            'pedosphere.decomposition.k_metabolic': 10.0,
            'pedosphere.decomposition.k_structural': 4.9,
            'pedosphere.decomposition.k_fast_som': 6.0,
            'pedosphere.decomposition.k_slow_som': 0.2,
            'pedosphere.decomposition.k_passive_som': 0.0045,
            'pedosphere.physical.q10_decomposition': 2.0,
            'pedosphere.decomposition.field_capacity': 0.25,
            'pedosphere.decomposition.wilting_point': 0.10,
            'pedosphere.state_defaults.clay_content': 0.25,
            'pedosphere.decomposition.lignin_n_threshold': 0.45,
            'pedosphere.decomposition.temp_response_type': 'q10',
            'pedosphere.decomposition.moisture_response_power': 1.0,
        }
        # Parameters the database does not define are registered by this test.
        test_only_specs = [
            ParameterSpec(key='pedosphere.carbon_pools.pool_names',
                          value=['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som'],
                          units='dimensionless', dtype='list', bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.carbon_pools.mass_balance_tolerance', value=1e-6, units='1', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.carbon_pools.min_pool_size', value=1e-8, units='g C m-2', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.carbon_pools.max_pool_size', value=1e3, units='g C m-2', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.transfer_matrix.t_met_fast', value=0.45, units='1', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.transfer_matrix.t_met_slow', value=0.45, units='1', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.transfer_matrix.t_str_fast', value=0.40, units='1', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.transfer_matrix.t_str_slow', value=0.40, units='1', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.transfer_matrix.t_fast_slow', value=0.42, units='1', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.transfer_matrix.t_fast_passive', value=0.03, units='1', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.transfer_matrix.t_slow_fast', value=0.45, units='1', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
            ParameterSpec(key='pedosphere.transfer_matrix.t_slow_passive', value=0.03, units='1', dtype="float32", bounds=(-float("inf"), float("inf")), category="model_parameter"),
        ]

        for key, value in test_params.items():
            manager.set_value(key, value, validate=False)
        for spec in test_only_specs:
            manager.register_parameter(spec)

        return manager

    @pytest.fixture
    def integrated_state(self):
        """Standard integrated state for testing."""
        return {
            'metabolic_litter_c': torch.tensor(0.1),      # g C m-2
            'structural_litter_c': torch.tensor(0.3),     # g C m-2
            'fast_som_c': torch.tensor(0.8),              # g C m-2
            'slow_som_c': torch.tensor(2.0),              # g C m-2
            'passive_som_c': torch.tensor(4.0),           # g C m-2
            'soil_temperature': torch.tensor(288.15),     # K
            'soil_moisture': torch.tensor(0.20),          # m3 m-3
            'metabolic_litter_input': torch.tensor(0.0),  # g C m-2 s-1
            'structural_litter_input': torch.tensor(0.0), # g C m-2 s-1
            'lignin_n_ratio': torch.tensor(0.3),          # dimensionless
            'clay_content': torch.tensor(0.25),           # dimensionless
        }
    
    # =============================================================================
    # L2.1: Decomposition-Carbon Pool Integration Tests
    # =============================================================================
    
    def test_decomposition_closures_are_evaluated_independently(
        self, parameter_manager, integrated_state
    ):
        """Test both decomposition closures without chaining their tendencies."""
        # Extract configurations - use list() and get() instead of get_parameter_group()
        decomp_param_keys = parameter_manager.list(namespace='pedosphere.decomposition')
        decomp_config = {key.split('.')[-1]: parameter_manager.get(key) for key in decomp_param_keys}
        pools_param_keys = parameter_manager.list(namespace='pedosphere.carbon_pools')
        pools_config = {key.split('.')[-1]: parameter_manager.get(key) for key in pools_param_keys}
        
        # Create integrated modules
        decomposer = FirstOrderDecomposition(
            name="integrated_decomp",
            config=decomp_config
        )
        
        pools_matrix = PoolsMatrix(
            name="integrated_pools", 
            config=pools_config
        )
        
        # Test decomposition rate calculation
        decomp_rates = call_forward(decomposer, integrated_state)
        
        # Verify FirstOrder exposes canonical pool-resolved gross flux diagnostics.
        expected_rate_keys = [
            'diagnostics.first_order_decomposition.metabolic_litter_decomposition_flux',
            'diagnostics.first_order_decomposition.structural_litter_decomposition_flux',
            'diagnostics.first_order_decomposition.fast_som_decomposition_flux',
            'diagnostics.first_order_decomposition.slow_som_decomposition_flux',
            'diagnostics.first_order_decomposition.passive_som_decomposition_flux',
        ]
        
        for key in expected_rate_keys:
            assert key in decomp_rates, f"Missing decomposition rate: {key}"
            assert torch.is_tensor(decomp_rates[key])
            assert torch.isfinite(decomp_rates[key]).all()
            assert decomp_rates[key] >= 0  # Rates should be non-negative

        assert decomp_rates['carbon_decomposition_rate'] > 0
        assert decomp_rates['heterotrophic_respiration'] > 0

        pool_keys = [
            'metabolic_litter_c',
            'structural_litter_c',
            'fast_som_c',
            'slow_som_c',
            'passive_som_c',
        ]
        # PoolsMatrix is an alternative closure and receives explicit ports.
        matrix_state = {key: integrated_state[key] for key in pool_keys}
        matrix_state['environmental_modifier'] = torch.ones_like(
            integrated_state['metabolic_litter_c']
        )
        for key in (
            'metabolic_litter_input',
            'structural_litter_input',
            'fast_som_input',
            'slow_som_input',
            'passive_som_input',
        ):
            matrix_state[key] = torch.zeros_like(
                integrated_state['metabolic_litter_c']
            )

        matrix_outputs = pools_matrix.forward(
            matrix_state,
            time=0.0,
            dt=3600.0,
        )
        for key in pool_keys:
            assert key in matrix_outputs
            assert isinstance(matrix_outputs[key], UpdateItem)
            assert matrix_outputs[key].mode == UpdateMode.ACCUMULATE
            assert matrix_outputs[key].per_timestep is True
            assert torch.isfinite(matrix_outputs[key].value).all()

        for key in ('carbon_decomposition_rate', 'heterotrophic_respiration'):
            assert isinstance(matrix_outputs[key], UpdateItem)
            assert matrix_outputs[key].mode == UpdateMode.REPLACE
            assert matrix_outputs[key].per_timestep is False
            assert torch.isfinite(matrix_outputs[key].value).all()
    
    def test_parameter_consistency_across_modules(self, parameter_manager):
        """Test parameter consistency between decomposition and pools modules."""
        # Get configurations
        decomp_param_keys = parameter_manager.list(namespace='pedosphere.decomposition')
        decomp_config = {key.split('.')[-1]: parameter_manager.get(key) for key in decomp_param_keys}
        pools_param_keys = parameter_manager.list(namespace='pedosphere.carbon_pools')
        pools_config = {key.split('.')[-1]: parameter_manager.get(key) for key in pools_param_keys}
        transfer_param_keys = parameter_manager.list(namespace='pedosphere.transfer_matrix')
        transfer_config = {key.split('.')[-1]: parameter_manager.get(key) for key in transfer_param_keys}
        
        # Create modules
        decomposer = FirstOrderDecomposition(name="test_decomp", config=decomp_config)
        pools_matrix = PoolsMatrix(name="test_pools", config=pools_config)
        
        # Check pool name consistency
        decomp_pools = set(decomposer.pool_params.keys())
        matrix_pools = set(pools_matrix.pool_names)
        
        assert decomp_pools == matrix_pools, \
            f"Pool name mismatch: {decomp_pools} vs {matrix_pools}"
        
        # Check transfer parameter consistency
        decomp_transfers = decomposer.transfer_params
        matrix_transfer_keys = [
            'metabolic_to_fast', 'metabolic_to_slow', 'structural_to_fast', 
            'structural_to_slow', 'fast_to_slow', 'fast_to_passive',
            'slow_to_fast', 'slow_to_passive'
        ]
        
        for key in matrix_transfer_keys:
            # Check that transfer parameters exist in both systems
            assert key in decomp_transfers, f"Missing transfer parameter: {key}"
            # Verify reasonable ranges
            assert 0.0 <= decomp_transfers[key] <= 1.0, f"Transfer {key} outside [0,1]"
    
    # =============================================================================
    # L2.2: Parameter Management Integration
    # =============================================================================
    
    def test_yaml_config_loading(self, temp_config_file):
        """Test loading parameters from YAML configuration."""
        from relic.parameters import get_global_manager, reset_global_manager
        
        # Reset and get global manager
        reset_global_manager()
        param_manager = get_global_manager()
        
        # Set test parameters with proper namespacing
        param_manager.set_value('pedosphere.decomposition.k_metabolic', 10.0, validate=False)
        param_manager.set_value('pedosphere.state_defaults.clay_content', 0.25, validate=False)
        param_manager.set_value('pedosphere.physical.q10_decomposition', 2.0, validate=False)
        param_manager.set_value('pedosphere.decomposition.temp_response_type', 'q10', validate=False) 
        param_manager.register_parameter(ParameterSpec(
            key='pedosphere.carbon_pools.pool_names',
            value=['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som'],
            units='dimensionless',
            dtype='list', bounds=(-float("inf"), float("inf")), category="model_parameter",
        ))

        # Test parameter retrieval
        k_metabolic = param_manager.get('pedosphere.decomposition.k_metabolic')
        assert k_metabolic == 10.0
        
        clay_content = param_manager.get('pedosphere.state_defaults.clay_content')
        assert clay_content == 0.25
        
        pool_names = param_manager.get('pedosphere.carbon_pools.pool_names')
        expected_pools = ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']
        assert pool_names == expected_pools
        
        # Test parameter groups
        decomp_param_keys = param_manager.list(namespace='pedosphere.decomposition')
        decomp_group = {key.split('.')[-1]: param_manager.get(key) for key in decomp_param_keys}
        assert 'k_metabolic' in decomp_group
        assert param_manager.get('pedosphere.physical.q10_decomposition') == 2.0
        assert 'temp_response_type' in decomp_group
        
    def test_module_parameter_integration(self, parameter_manager):
        """Test integration between parameter manager and modules."""
        # Get configuration from parameter manager
        param_keys = parameter_manager.list(namespace='pedosphere.decomposition')
        config = {key.split('.')[-1]: parameter_manager.get(key) for key in param_keys}
        
        # Create module with parameters
        decomposer = FirstOrderDecomposition(name="param_test", config=config)
        
        # Verify parameters were loaded correctly
        assert decomposer.get_parameter_value('k_metabolic') == 10.0
        assert decomposer.get_parameter_value(
            'q10_decomposition',
            namespace='pedosphere.physical',
            required=True,
        ) == 2.0
        assert decomposer.get_parameter_value(
            'clay_content',
            namespace='pedosphere.state_defaults',
        ) == 0.25
        
        # Test parameter usage in calculations
        pool_params = decomposer.pool_params
        assert 'metabolic_litter' in pool_params
        
        # Check k_ref was correctly converted from per-year to per-second
        k_ref = pool_params['metabolic_litter']['k_ref']
        seconds_per_year = float(get_physical_constant('seconds_per_year'))
        expected_k_ref = 10.0 / seconds_per_year
        assert abs(k_ref - expected_k_ref) < 1e-10
    
    # =============================================================================
    # L2.3: Environmental Driver Integration  
    # =============================================================================
    
    def test_temperature_moisture_coupling(self, parameter_manager, integrated_state):
        """Test coupling of temperature and moisture responses."""
        param_keys = parameter_manager.list(namespace='pedosphere.decomposition')
        config = {key.split('.')[-1]: parameter_manager.get(key) for key in param_keys}
        decomposer = FirstOrderDecomposition(name="env_test", config=config)
        
        # Test individual responses
        temp_response = decomposer._calculate_temperature_response(
            integrated_state['soil_temperature']
        )
        moisture_response = decomposer._calculate_moisture_response(
            integrated_state['soil_moisture']
        )
        
        # Test coupled calculation
        rates = call_forward(decomposer, integrated_state)
        
        # Verify responses are properly coupled in rate calculation
        # Extract effective rate for metabolic litter
        metabolic_rate = rates[
            'diagnostics.first_order_decomposition.metabolic_litter_decomposition_flux'
        ]
        metabolic_pool = integrated_state['metabolic_litter_c']
        
        # Calculate expected rate components. The metabolic pool is the
        # lignin-free litter fraction and is not lignin inhibited (Parton et
        # al. 1987, Eq. [2]-[4] apply the lignin term to structural litter).
        k_ref = decomposer.pool_params['metabolic_litter']['k_ref']
        expected_env_modifier = temp_response * moisture_response
        expected_rate = k_ref * expected_env_modifier * metabolic_pool
        
        # Allow for small numerical differences
        relative_error = abs(float(metabolic_rate - expected_rate)) / float(expected_rate + 1e-10)
        assert relative_error < 0.05, f"Environmental coupling error: {relative_error:.4f}"
    
    def test_environmental_sensitivity_integration(self, parameter_manager):
        """Test integrated environmental sensitivity across temperature and moisture."""
        param_keys = parameter_manager.list(namespace='pedosphere.decomposition')
        config = {key.split('.')[-1]: parameter_manager.get(key) for key in param_keys}
        decomposer = FirstOrderDecomposition(name="sens_test", config=config)
        
        # Test matrix of environmental conditions
        temperatures = torch.tensor([278.15, 288.15, 298.15])  # 5 C, 15 C, 25 C
        moistures = torch.tensor([0.10, 0.20, 0.30])           # 10%, 20%, 30%
        
        base_state = {
            'metabolic_litter_c': torch.tensor(0.1),
            'structural_litter_c': torch.tensor(0.3),
            'fast_som_c': torch.tensor(0.8),
            'slow_som_c': torch.tensor(2.0), 
            'passive_som_c': torch.tensor(4.0),
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
            'lignin_n_ratio': torch.tensor(0.3),
            'clay_content': torch.tensor(0.25),
        }
        
        response_matrix = []
        
        for temp in temperatures:
            temp_responses = []
            for moisture in moistures:
                test_state = base_state.copy()
                test_state['soil_temperature'] = temp
                test_state['soil_moisture'] = moisture
                
                rates = call_forward(decomposer, test_state)
                total_rate = float(rates['carbon_decomposition_rate'])
                temp_responses.append(total_rate)
            
            response_matrix.append(temp_responses)
        
        # Verify expected patterns
        response_matrix = np.array(response_matrix)
        
        # Check temperature sensitivity (rates should generally increase with temperature)
        for moisture_idx in range(len(moistures)):
            temp_series = response_matrix[:, moisture_idx]
            # Most rates should increase with temperature (some may level off)
            increasing_pairs = sum(1 for i in range(len(temp_series)-1) 
                                 if temp_series[i+1] >= temp_series[i])
            assert increasing_pairs >= len(temp_series) // 2, \
                "Temperature sensitivity pattern broken"
        
        # Check that responses are positive and finite
        assert np.all(np.isfinite(response_matrix))
        assert np.all(response_matrix >= 0)
    
    # =============================================================================
    # L2.4: Mass Balance Integration
    # =============================================================================
    
    def test_integrated_mass_balance(self, parameter_manager, integrated_state):
        """Test the FirstOrder carbon ledger without a second closure."""
        decomp_param_keys = parameter_manager.list(namespace='pedosphere.decomposition')
        decomp_config = {key.split('.')[-1]: parameter_manager.get(key) for key in decomp_param_keys}
        decomposer = FirstOrderDecomposition(name="balance_decomp", config=decomp_config)

        dt = 3600.0
        outputs = call_forward(decomposer, integrated_state, dt=dt)
        pool_keys = [
            'metabolic_litter_c',
            'structural_litter_c',
            'fast_som_c',
            'slow_som_c',
            'passive_som_c',
        ]
        residual = sum(outputs[key] for key in pool_keys)
        residual = residual + outputs['heterotrophic_respiration'] * dt
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-9, rtol=0.0)
    
    def test_conservation_with_inputs(self, parameter_manager, integrated_state):
        """Test mass conservation with external litter inputs."""
        decomp_param_keys = parameter_manager.list(namespace='pedosphere.decomposition')
        decomp_config = {key.split('.')[-1]: parameter_manager.get(key) for key in decomp_param_keys}
        
        decomposer = FirstOrderDecomposition(name="input_test", config=decomp_config)
        
        metabolic_input = torch.tensor(8.5e-7)  # g C m-2 s-1
        structural_input = torch.tensor(1.5e-7)  # g C m-2 s-1

        state = dict(integrated_state)
        state['metabolic_litter_input'] = metabolic_input
        state['structural_litter_input'] = structural_input
        dt = 3600.0
        outputs = call_forward(decomposer, state, dt=dt)
        pool_keys = [
            'metabolic_litter_c',
            'structural_litter_c',
            'fast_som_c',
            'slow_som_c',
            'passive_som_c',
        ]
        residual = sum(outputs[key] for key in pool_keys)
        residual = (
            residual
            + outputs['heterotrophic_respiration'] * dt
            - (metabolic_input + structural_input) * dt
        )
        assert torch.allclose(residual, torch.zeros_like(residual), atol=2e-9, rtol=0.0)

    def test_litter_cn_route_simulator_state_manager_provenance(self, caplog):
        """Test the phenology-to-soil C/N route through StateManager.

        References:
        - Thornton et al. (2002), AFM 113, 185-222.
        - Parton et al. (1987), SSSAJ 51, 1173-1179.
        - Manzoni and Porporato (2009), SBB 41, 1355-1367.

        Mathematical expressions:
        - F_soil = 1000 * F_vegetation.
        - Delta C_veg,g + Delta C_soil + R_h * dt = 0.
        - Delta N_veg,g + Delta N_soil = 0.

        Parameter meanings and units:
        - dt is the step duration [s].
        - F is a boundary flux [kg element m-2 s-1] or [g element m-2 s-1].
        - Delta C and Delta N are StateManager increments [kg or g element m-2].
        """
        pool_keys = (
            "metabolic_litter_c",
            "structural_litter_c",
            "fast_som_c",
            "slow_som_c",
            "passive_som_c",
        )
        nitrogen_pool_keys = (
            "organic_nitrogen",
            "nh4_nitrogen",
            "microbial_nitrogen",
        )
        vegetation_carbon_keys = ("leaf_carbon", "froot_carbon")
        vegetation_nitrogen_keys = (
            "leaf_nitrogen",
            "froot_nitrogen",
            "plant_nitrogen_pool",
        )
        state = {
            "metabolic_litter_c": torch.tensor(100.0, dtype=torch.float64),
            "structural_litter_c": torch.tensor(200.0, dtype=torch.float64),
            "fast_som_c": torch.tensor(500.0, dtype=torch.float64),
            "slow_som_c": torch.tensor(1000.0, dtype=torch.float64),
            "passive_som_c": torch.tensor(2000.0, dtype=torch.float64),
            "soil_temperature": torch.tensor(293.15, dtype=torch.float64),
            "soil_moisture": torch.tensor(0.3, dtype=torch.float64),
            "day_of_year": torch.tensor(280.0, dtype=torch.float64),
            "air_temperature": torch.tensor(283.15, dtype=torch.float64),
            "leaf_carbon": torch.tensor(2.0, dtype=torch.float64),
            "froot_carbon": torch.tensor(1.5, dtype=torch.float64),
            "leaf_nitrogen": torch.tensor(0.08, dtype=torch.float64),
            "froot_nitrogen": torch.tensor(0.04, dtype=torch.float64),
            "plant_nitrogen_pool": torch.tensor(0.2, dtype=torch.float64),
            "lignin_n_ratio": torch.tensor(0.3, dtype=torch.float64),
            "clay_content": torch.tensor(0.25, dtype=torch.float64),
            "organic_nitrogen": torch.tensor(5.0, dtype=torch.float64),
            "nh4_nitrogen": torch.tensor(0.1, dtype=torch.float64),
            "microbial_nitrogen": torch.tensor(0.05, dtype=torch.float64),
            "soil_ph": torch.tensor(6.5, dtype=torch.float64),
        }
        dt = 3600.0
        phenology = StaticPhenology(name="static_phenology")
        phenology_updates = phenology.forward(
            {key: value.clone() for key, value in state.items()},
            time=0.0,
            dt=dt,
        )
        mapped_state = {key: value.clone() for key, value in state.items()}
        for key in (
            "leaf_litterfall_carbon",
            "froot_litterfall_carbon",
            "leaf_litterfall_nitrogen",
            "froot_litterfall_nitrogen",
        ):
            mapped_state[key] = phenology_updates[key].value.clone()
        mapper = LitterfallInterfaceMapper(
            name="litterfall_interface_mapper",
            config={"dependencies": ["static_phenology"]},
        )
        mapped = mapper.forward(
            mapped_state,
            time=0.0,
            dt=dt,
        )
        direct_state = {key: value.clone() for key, value in state.items()}
        for key in ("metabolic_litter_input", "structural_litter_input"):
            direct_state[key] = mapped[key].value.clone()
        direct_functor = FirstOrderDecomposition(
            name="first_order_decomposition",
            config={"dependencies": ["litterfall_interface_mapper"]},
        )
        direct = direct_functor.forward(
            direct_state,
            time=0.0,
            dt=dt,
        )
        direct_state["carbon_decomposition_rate"] = direct["carbon_decomposition_rate"].value.clone()
        direct_state["organic_nitrogen_input"] = mapped["organic_nitrogen_input"].value.clone()
        direct_mineralization = MineralizationProcess(
            name="mineralization",
            config={
                "dependencies": [
                    "first_order_decomposition",
                    "litterfall_interface_mapper",
                ]
            },
        )
        direct_nitrogen = direct_mineralization.forward(
            direct_state,
            time=0.0,
            dt=dt,
        )

        metabolic_input = mapped["metabolic_litter_input"]
        structural_input = mapped["structural_litter_input"]
        organic_nitrogen_input = mapped["organic_nitrogen_input"]
        total_input = metabolic_input.value + structural_input.value
        phenology_carbon_flux = (
            mapped_state["leaf_litterfall_carbon"]
            + mapped_state["froot_litterfall_carbon"]
        )
        phenology_nitrogen_flux = (
            mapped_state["leaf_litterfall_nitrogen"]
            + mapped_state["froot_litterfall_nitrogen"]
        )
        torch.testing.assert_close(
            total_input,
            phenology_carbon_flux * 1000.0,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            metabolic_input.value / total_input,
            torch.as_tensor(
                mapper.max_metabolic_fraction,
                dtype=total_input.dtype,
                device=total_input.device,
            ),
            rtol=2.0 * torch.finfo(total_input.dtype).eps,
            atol=0.0,
        )
        assert mapper.max_metabolic_fraction == pytest.approx(0.9, rel=1.0e-7)
        assert metabolic_input.unit == "g C m-2 s-1"
        assert structural_input.unit == "g C m-2 s-1"
        assert metabolic_input.mode == UpdateMode.REPLACE
        assert structural_input.mode == UpdateMode.REPLACE
        assert metabolic_input.per_timestep is False
        assert structural_input.per_timestep is False
        torch.testing.assert_close(
            organic_nitrogen_input.value,
            phenology_nitrogen_flux * 1000.0,
            rtol=0.0,
            atol=0.0,
        )
        assert organic_nitrogen_input.unit == "g N m-2 s-1"
        assert organic_nitrogen_input.mode == UpdateMode.REPLACE
        assert organic_nitrogen_input.per_timestep is False

        state_manager = StateManager(
            device="cpu",
            conservation_checks={"enabled": False},
        )
        state_manager.set_state(
            {key: value.clone() for key, value in state.items()},
            validate=True,
        )
        simulator = RELICSimulator(
            {
                "device": "cpu",
                "time_step": dt,
                "max_time": dt,
                "conservation_checks": False,
                "strict_dependencies": True,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize(state_manager, DependencyGraphResolver())
        simulator.register_functor(
            StaticPhenology(name="static_phenology"),
            execution_group="fast",
        )
        simulator.register_functor(
            LitterfallInterfaceMapper(
                name="litterfall_interface_mapper",
                config={"dependencies": ["static_phenology"]},
            ),
            execution_group="fast",
        )
        simulator.register_functor(
            FirstOrderDecomposition(
                name="first_order_decomposition",
                config={"dependencies": ["litterfall_interface_mapper"]},
            ),
            execution_group="fast",
        )
        simulator.register_functor(
            MineralizationProcess(
                name="mineralization",
                config={
                    "dependencies": [
                        "first_order_decomposition",
                        "litterfall_interface_mapper",
                    ]
                },
            ),
            execution_group="fast",
        )
        initial = {
            key: state_manager.get_state(key).clone()
            for key in (
                *pool_keys,
                *nitrogen_pool_keys,
                *vegetation_carbon_keys,
                *vegetation_nitrogen_keys,
            )
        }

        caplog.clear()
        with caplog.at_level("WARNING"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                step_updates = simulator.step(dt=dt)

        assert set(mapped).issubset(step_updates)
        assert set(direct).issubset(step_updates)
        assert set(direct_nitrogen).issubset(step_updates)
        user_warnings = [
            str(warning.message)
            for warning in caught
            if issubclass(warning.category, UserWarning)
        ]
        assert not user_warnings, user_warnings
        assert not caplog.records, [record.getMessage() for record in caplog.records]
        execution_order = simulator.dependency_graph.topological_sort()
        phenology_index = execution_order.index("static_phenology")
        mapper_index = execution_order.index("litterfall_interface_mapper")
        first_order_index = execution_order.index("first_order_decomposition")
        mineralization_index = execution_order.index("mineralization")
        assert phenology_index < mapper_index < first_order_index < mineralization_index
        for key in (
            "leaf_litterfall_carbon",
            "froot_litterfall_carbon",
            "leaf_litterfall_nitrogen",
            "froot_litterfall_nitrogen",
        ):
            actual_flux = state_manager.get_state(key)
            torch.testing.assert_close(
                actual_flux,
                phenology_updates[key].value.to(
                    device=actual_flux.device,
                    dtype=actual_flux.dtype,
                ),
                rtol=0.0,
                atol=0.0,
            )
            provenance_writers = {
                record["functor"]
                for record in simulator.get_variable_provenance(key)
            }
            assert provenance_writers == {"static_phenology"}
        for key in (*vegetation_carbon_keys, *vegetation_nitrogen_keys):
            item = phenology_updates[key]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            simulator_delta = step_updates[key]
            final_state = state_manager.get_state(key)
            torch.testing.assert_close(
                final_state,
                initial[key]
                + simulator_delta.to(
                    device=initial[key].device,
                    dtype=initial[key].dtype,
                ),
                rtol=0.0,
                atol=0.0,
            )
            provenance_writers = {
                record["functor"]
                for record in simulator.get_variable_provenance(key)
            }
            assert provenance_writers == {"static_phenology"}
        for key in (
            "metabolic_litter_input",
            "structural_litter_input",
            "organic_nitrogen_input",
        ):
            actual_port = state_manager.get_state(key)
            torch.testing.assert_close(
                actual_port,
                mapped[key].value.to(
                    device=actual_port.device,
                    dtype=actual_port.dtype,
                ),
                rtol=0.0,
                atol=0.0,
            )
            provenance_writers = {
                record["functor"]
                for record in simulator.get_variable_provenance(key)
            }
            assert provenance_writers == {"litterfall_interface_mapper"}

        for key in pool_keys:
            item = direct[key]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            simulator_delta = step_updates[key]
            assert isinstance(simulator_delta, torch.Tensor)
            final_state = state_manager.get_state(key)
            expected_final = initial[key] + simulator_delta.to(
                device=initial[key].device,
                dtype=initial[key].dtype,
            )
            torch.testing.assert_close(
                final_state,
                expected_final,
                rtol=0.0,
                atol=0.0,
            )
            actual_delta = final_state - initial[key]
            roundoff_bound = (
                8.0
                * torch.finfo(actual_delta.dtype).eps
                * torch.maximum(torch.abs(initial[key]), torch.ones_like(initial[key]))
            )
            assert torch.all(
                torch.abs(
                    actual_delta
                    - simulator_delta.to(
                        device=actual_delta.device,
                        dtype=actual_delta.dtype,
                    )
                )
                <= roundoff_bound
            )
            provenance_writers = {
                record["functor"]
                for record in simulator.get_variable_provenance(key)
            }
            assert provenance_writers == {"first_order_decomposition"}

        torch.testing.assert_close(
            step_updates["carbon_decomposition_rate"],
            direct["carbon_decomposition_rate"].value.to(dtype=step_updates["carbon_decomposition_rate"].dtype),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            step_updates["organic_nitrogen_input"],
            organic_nitrogen_input.value.to(dtype=step_updates["organic_nitrogen_input"].dtype),
            rtol=0.0,
            atol=0.0,
        )
        for key in nitrogen_pool_keys:
            item = direct_nitrogen[key]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            simulator_delta = step_updates[key]
            assert isinstance(simulator_delta, torch.Tensor)
            final_state = state_manager.get_state(key)
            expected_final = initial[key] + simulator_delta.to(
                device=initial[key].device,
                dtype=initial[key].dtype,
            )
            torch.testing.assert_close(
                final_state,
                expected_final,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                simulator_delta,
                item.value.to(
                    device=simulator_delta.device,
                    dtype=simulator_delta.dtype,
                ),
                rtol=1.0e-12,
                atol=1.0e-15,
            )
            provenance_writers = {
                record["functor"]
                for record in simulator.get_variable_provenance(key)
            }
            assert provenance_writers == {"mineralization"}

        nitrogen_state_delta = torch.stack(
            [
                state_manager.get_state(key) - initial[key]
                for key in nitrogen_pool_keys
            ]
        ).sum()
        nitrogen_boundary_step = step_updates[
            "organic_nitrogen_input"
        ].to(dtype=nitrogen_state_delta.dtype) * torch.as_tensor(
            dt,
            dtype=nitrogen_state_delta.dtype,
        )
        torch.testing.assert_close(
            nitrogen_state_delta,
            nitrogen_boundary_step,
            rtol=0.0,
            atol=1.0e-14,
        )
        boundary_diagnostic = step_updates[
            "diagnostics.mineralization.organic_nitrogen_boundary_input"
        ]
        torch.testing.assert_close(
            boundary_diagnostic,
            direct_nitrogen[
                "diagnostics.mineralization.organic_nitrogen_boundary_input"
            ].value.to(dtype=boundary_diagnostic.dtype),
            rtol=0.0,
            atol=0.0,
        )
        nitrogen_violations = direct_mineralization.check_conservation_laws(
            direct_state,
            direct_nitrogen,
        )
        assert nitrogen_violations["nitrogen"] <= 1.0e-14
        assert nitrogen_violations["mass"] <= 1.0e-14

        vegetation_carbon_delta_g = 1000.0 * torch.stack(
            [
                state_manager.get_state(key) - initial[key]
                for key in vegetation_carbon_keys
            ]
        ).sum()
        pedosphere_carbon_delta = torch.stack(
            [
                state_manager.get_state(key) - initial[key]
                for key in pool_keys
            ]
        ).sum()
        respiration_step = step_updates["heterotrophic_respiration"].to(
            dtype=pedosphere_carbon_delta.dtype
        ) * torch.as_tensor(dt, dtype=pedosphere_carbon_delta.dtype)
        cross_domain_carbon_residual = (
            vegetation_carbon_delta_g
            + pedosphere_carbon_delta
            + respiration_step
        )

        vegetation_nitrogen_delta_g = 1000.0 * torch.stack(
            [
                state_manager.get_state(key) - initial[key]
                for key in vegetation_nitrogen_keys
            ]
        ).sum()
        cross_domain_nitrogen_residual = (
            vegetation_nitrogen_delta_g + nitrogen_state_delta
        )
        float32_rounding = torch.finfo(torch.float32).eps
        carbon_scale = torch.stack(
            [
                torch.abs(vegetation_carbon_delta_g),
                torch.abs(pedosphere_carbon_delta),
                torch.abs(respiration_step),
            ]
        ).max()
        nitrogen_scale = torch.stack(
            [
                torch.abs(vegetation_nitrogen_delta_g),
                torch.abs(nitrogen_state_delta),
            ]
        ).max()
        assert torch.abs(cross_domain_carbon_residual) <= (
            16.0 * float32_rounding * carbon_scale
        )
        assert torch.abs(cross_domain_nitrogen_residual) <= (
            16.0 * float32_rounding * nitrogen_scale
        )

        violations = direct_functor.check_conservation_laws(
            direct_state,
            direct,
            dt=dt,
        )
        pool_delta = torch.stack([direct[key].value for key in pool_keys]).sum()
        respiration_step = direct["heterotrophic_respiration"].value * dt
        input_step = total_input.to(dtype=pool_delta.dtype) * dt
        direct_carbon_residual = pool_delta + respiration_step - input_step
        direct_carbon_scale = torch.stack(
            [
                torch.abs(pool_delta),
                torch.abs(respiration_step),
                torch.abs(input_step),
            ]
        ).max()
        direct_carbon_bound = 16.0 * float32_rounding * direct_carbon_scale
        assert violations["carbon"] <= float(
            direct_carbon_bound.detach().item()
        )
        assert torch.abs(direct_carbon_residual) <= direct_carbon_bound
        assert simulator.get_effective_device("static_phenology") == torch.device("cpu")
        assert simulator.get_effective_device("litterfall_interface_mapper") == torch.device("cpu")
        assert simulator.get_effective_device("first_order_decomposition") == torch.device("cpu")
        assert simulator.get_effective_device("mineralization") == torch.device("cpu")

    def test_grazing_excreta_routes_reach_carbon_and_nitrogen_ledgers(
        self,
        caplog,
    ):
        """Test dependency-selected grazing dung and urine boundary routes.

        References:
        - Coleman and Jenkinson (1996), RothC-26.3 model guide.
        - Li et al. (1992), DNDC manure carbon-nitrogen routing.
        Mathematical expressions:
        - F_met,dung = f_man * convert(F_dung,C).
        - F_str,dung = (1 - f_man) * convert(F_dung,C).
        - F_organic,N = convert(F_dung,N).
        - Delta_NH4,urine = convert(F_urine,N) * dt.
        Parameter meanings:
        - f_man is the non-strict two-pool manure projection fraction [1].
        - All source fluxes use kg element m-2 s-1 and target fluxes use
          g element m-2 s-1.
        """
        dt = 3600.0
        state = {
            "aboveground_biomass": torch.tensor(3.0, dtype=torch.float64),
            "vegetation_quality": torch.tensor(0.8, dtype=torch.float64),
            "soil_moisture": torch.tensor(0.3, dtype=torch.float64),
            "day_of_year": torch.tensor(150.0, dtype=torch.float64),
            "cumulative_intake": torch.tensor(0.0, dtype=torch.float64),
            "days_grazed": torch.tensor(0.0, dtype=torch.float64),
            "leaf_litterfall_carbon": torch.tensor(0.0, dtype=torch.float64),
            "froot_litterfall_carbon": torch.tensor(0.0, dtype=torch.float64),
            "leaf_litterfall_nitrogen": torch.tensor(0.0, dtype=torch.float64),
            "froot_litterfall_nitrogen": torch.tensor(0.0, dtype=torch.float64),
            "lignin_n_ratio": torch.tensor(0.3, dtype=torch.float64),
            "metabolic_litter_c": torch.tensor(100.0, dtype=torch.float64),
            "structural_litter_c": torch.tensor(200.0, dtype=torch.float64),
            "fast_som_c": torch.tensor(500.0, dtype=torch.float64),
            "slow_som_c": torch.tensor(1000.0, dtype=torch.float64),
            "passive_som_c": torch.tensor(2000.0, dtype=torch.float64),
            "soil_temperature": torch.tensor(293.15, dtype=torch.float64),
            "clay_content": torch.tensor(0.25, dtype=torch.float64),
            "organic_nitrogen": torch.tensor(5.0, dtype=torch.float64),
            "nh4_nitrogen": torch.tensor(0.1, dtype=torch.float64),
            "microbial_nitrogen": torch.tensor(0.05, dtype=torch.float64),
            "soil_ph": torch.tensor(6.5, dtype=torch.float64),
        }
        grazing = GrazingManagement(
            name="grazing_management",
            config={"calendar_type": "noleap"},
        )
        grazing_updates = grazing.forward(
            {key: value.clone() for key, value in state.items()},
            time=0.0,
            dt=dt,
        )
        mapped_state = {key: value.clone() for key, value in state.items()}
        for key in ("dung_c_flux", "dung_n_flux"):
            mapped_state[key] = grazing_updates[key].value.clone()
        mapper = LitterfallInterfaceMapper(
            name="litterfall_interface_mapper",
            config={"dependencies": ["grazing_management"]},
        )
        mapped = mapper.forward(mapped_state, time=0.0, dt=dt)

        expected_dung_carbon = grazing_updates["dung_c_flux"].value * 1000.0
        expected_dung_nitrogen = grazing_updates["dung_n_flux"].value * 1000.0
        torch.testing.assert_close(
            mapped["metabolic_litter_input"].value,
            mapper.manure_metabolic_fraction * expected_dung_carbon,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            mapped["structural_litter_input"].value,
            (1.0 - mapper.manure_metabolic_fraction) * expected_dung_carbon,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            mapped["organic_nitrogen_input"].value,
            expected_dung_nitrogen,
            rtol=0.0,
            atol=0.0,
        )
        assert mapper.check_conservation_laws(mapped_state, mapped) == {
            "carbon": 0.0,
            "nitrogen": 0.0,
        }
        nitrogen_router = NitrogenCycling(
            config={"dependencies": ["grazing_management"]},
            enable_mineralization=False,
            enable_nitrification=False,
            enable_denitrification=False,
            enable_leaching=False,
            enable_root_uptake=False,
        )
        urine_state = {
            "urine_n_flux": grazing_updates["urine_n_flux"].value.clone(),
        }
        urine_route = nitrogen_router.forward(
            urine_state,
            time=0.0,
            dt=dt,
            subprocess_outputs={},
        )
        expected_urine_delta = (
            grazing_updates["urine_n_flux"].value * 1000.0 * dt
        )
        torch.testing.assert_close(
            urine_route["nh4_nitrogen"].value,
            expected_urine_delta,
            rtol=0.0,
            atol=0.0,
        )
        assert nitrogen_router.check_conservation_laws(
            urine_state,
            urine_route,
        ) == {"nitrogen": 0.0}

        state_manager = StateManager(
            device="cpu",
            conservation_checks={"enabled": False},
        )
        state_manager.set_state(
            {key: value.clone() for key, value in state.items()},
            validate=True,
        )
        simulator = RELICSimulator(
            {
                "device": "cpu",
                "time_step": dt,
                "max_time": dt,
                "conservation_checks": False,
                "strict_dependencies": True,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize(state_manager, DependencyGraphResolver())
        simulator.register_functor(
            GrazingManagement(name="grazing_management"),
            execution_group="fast",
        )
        simulator.register_functor(
            LitterfallInterfaceMapper(
                name="litterfall_interface_mapper",
                config={"dependencies": ["grazing_management"]},
            ),
            execution_group="fast",
        )
        simulator.register_functor(
            NitrogenCycling(
                config={"dependencies": ["grazing_management"]},
                enable_mineralization=False,
                enable_nitrification=False,
                enable_denitrification=False,
                enable_leaching=False,
                enable_root_uptake=False,
            ),
            execution_group="fast",
        )
        simulator.register_functor(
            FirstOrderDecomposition(
                name="first_order_decomposition",
                config={"dependencies": ["litterfall_interface_mapper"]},
            ),
            execution_group="fast",
        )
        simulator.register_functor(
            MineralizationProcess(
                name="mineralization",
                config={
                    "dependencies": [
                        "first_order_decomposition",
                        "litterfall_interface_mapper",
                    ]
                },
            ),
            execution_group="fast",
        )

        initial_carbon = torch.stack(
            [
                state[key]
                for key in (
                    "metabolic_litter_c",
                    "structural_litter_c",
                    "fast_som_c",
                    "slow_som_c",
                    "passive_som_c",
                )
            ]
        ).sum()
        initial_nitrogen = torch.stack(
            [
                state[key]
                for key in (
                    "organic_nitrogen",
                    "nh4_nitrogen",
                    "microbial_nitrogen",
                )
            ]
        ).sum()
        caplog.clear()
        with caplog.at_level("WARNING"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                updates = simulator.step(dt=dt)

        for key in (
            "dung_c_flux",
            "dung_n_flux",
            "urine_n_flux",
            "metabolic_litter_input",
            "structural_litter_input",
            "organic_nitrogen_input",
        ):
            writers = {
                record["functor"]
                for record in simulator.get_variable_provenance(key)
            }
            expected_writer = (
                "grazing_management"
                if key in {"dung_c_flux", "dung_n_flux", "urine_n_flux"}
                else "litterfall_interface_mapper"
            )
            assert writers == {expected_writer}
        assert (
            simulator.get_effective_device("litterfall_interface_mapper").type
            == "cpu"
        )
        assert simulator.get_effective_device("NitrogenCycling").type == "cpu"
        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, UserWarning)
        ]
        assert not caplog.records

        final_carbon = torch.stack(
            [
                state_manager.get_state(key)
                for key in (
                    "metabolic_litter_c",
                    "structural_litter_c",
                    "fast_som_c",
                    "slow_som_c",
                    "passive_som_c",
                )
            ]
        ).sum()
        respiration_step = updates["heterotrophic_respiration"].to(
            dtype=final_carbon.dtype
        ) * dt
        carbon_boundary_step = (
            updates["metabolic_litter_input"]
            + updates["structural_litter_input"]
        ).to(dtype=final_carbon.dtype) * dt
        carbon_residual = (
            final_carbon
            - initial_carbon
            + respiration_step
            - carbon_boundary_step
        )
        # The residual cancels O(pool) totals, so its rounding scales with the
        # pool magnitudes, not only with the step increments.
        carbon_scale = torch.stack(
            [
                torch.abs(final_carbon),
                torch.abs(initial_carbon),
                torch.abs(respiration_step),
                torch.abs(carbon_boundary_step),
            ]
        ).max()
        # The pools and the emitted inputs may carry different precisions;
        # the rounding bound follows the coarser of the two.
        rounding_eps = max(
            torch.finfo(updates["metabolic_litter_input"].dtype).eps,
            torch.finfo(final_carbon.dtype).eps,
        )
        carbon_roundoff_bound = 8.0 * rounding_eps * carbon_scale
        assert torch.abs(carbon_residual) <= carbon_roundoff_bound

        final_nitrogen = torch.stack(
            [
                state_manager.get_state(key)
                for key in (
                    "organic_nitrogen",
                    "nh4_nitrogen",
                    "microbial_nitrogen",
                )
            ]
        ).sum()
        nitrogen_boundary_step = (
            updates["organic_nitrogen_input"].to(
                dtype=final_nitrogen.dtype
            )
            + updates["urine_n_flux"].to(dtype=final_nitrogen.dtype) * 1000.0
        ) * dt
        nitrogen_residual = (
            final_nitrogen - initial_nitrogen - nitrogen_boundary_step
        )
        nitrogen_scale = torch.stack(
            [
                torch.abs(final_nitrogen),
                torch.abs(initial_nitrogen),
                torch.abs(nitrogen_boundary_step),
            ]
        ).max()
        nitrogen_roundoff_bound = (
            8.0
            * max(
                torch.finfo(updates["urine_n_flux"].dtype).eps,
                torch.finfo(final_nitrogen.dtype).eps,
            )
            * nitrogen_scale
        )
        assert torch.abs(nitrogen_residual) <= nitrogen_roundoff_bound
        nh4_writers = {
            record["functor"]
            for record in simulator.get_variable_provenance("nh4_nitrogen")
            if record.get("role") == "increment_source"
        }
        assert nh4_writers == {"NitrogenCycling", "mineralization"}

    def test_pools_matrix_simulator_state_manager_provenance(self):
        """Test PoolsMatrix through the canonical simulator lifecycle."""
        pool_keys = PoolsMatrix.POOL_STATE_KEYS
        state = {
            "metabolic_litter_c": torch.tensor(100.0, dtype=torch.float64),
            "structural_litter_c": torch.tensor(200.0, dtype=torch.float64),
            "fast_som_c": torch.tensor(500.0, dtype=torch.float64),
            "slow_som_c": torch.tensor(1000.0, dtype=torch.float64),
            "passive_som_c": torch.tensor(2000.0, dtype=torch.float64),
            "environmental_modifier": torch.tensor(1.0, dtype=torch.float64),
            "metabolic_litter_input": torch.tensor(1.0e-6, dtype=torch.float64),
            "structural_litter_input": torch.tensor(0.0, dtype=torch.float64),
            "fast_som_input": torch.tensor(0.0, dtype=torch.float64),
            "slow_som_input": torch.tensor(0.0, dtype=torch.float64),
            "passive_som_input": torch.tensor(0.0, dtype=torch.float64),
        }
        dt = 3600.0
        direct_functor = PoolsMatrix(name="pools_matrix", config={})
        direct = direct_functor.forward(
            {key: value.clone() for key, value in state.items()},
            time=0.0,
            dt=dt,
        )

        state_manager = StateManager(
            device="cpu",
            conservation_checks={"enabled": False},
        )
        state_manager.set_state(
            {key: value.clone() for key, value in state.items()},
            validate=True,
        )
        simulator = RELICSimulator(
            {
                "device": "cpu",
                "time_step": dt,
                "max_time": dt,
                "conservation_checks": False,
                "strict_dependencies": False,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize(state_manager, DependencyGraphResolver())
        simulator.register_functor(
            PoolsMatrix(name="pools_matrix", config={}),
            execution_group="fast",
        )
        initial = {
            key: state_manager.get_state(key).clone()
            for key in pool_keys
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step_updates = simulator.step(dt=dt)

        assert set(step_updates) == set(direct)
        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, UserWarning)
        ]
        for key in pool_keys:
            item = direct[key]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            actual_delta = state_manager.get_state(key) - initial[key]
            torch.testing.assert_close(
                actual_delta,
                item.value.to(dtype=actual_delta.dtype),
                rtol=0.0,
                atol=0.0,
            )
            provenance_writers = {
                record["functor"]
                for record in simulator.get_variable_provenance(key)
            }
            assert provenance_writers == {"pools_matrix"}

        residual = (
            sum(direct[key].value for key in pool_keys)
            + direct["heterotrophic_respiration"].value * dt
            - sum(state[key] for key in PoolsMatrix.BOUNDARY_INPUT_KEYS) * dt
        )
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            rtol=0.0,
            atol=2.0e-12,
        )
        assert simulator.get_effective_device("pools_matrix") == torch.device("cpu")
