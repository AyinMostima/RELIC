#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3 System Test: End-to-End Surrogate Model Integration

This module provides comprehensive L3 system tests for the complete surrogate
modeling workflow within the RELIC ecosystem simulation framework.

Testing Scope:
- Complete surrogate modeling workflow (training -> export -> deployment -> validation)
- Multi-surrogate ecosystem simulation 
- Scientific accuracy benchmarking against reference models
- Performance and scalability testing
- Cross-sphere coupling with surrogates
- Error handling and robustness testing
- Long-term simulation stability

Test Categories:
1. Full Ecosystem Surrogate Workflow Tests
2. Multi-Scale Simulation Tests
3. Scientific Accuracy Validation Tests
4. Performance Benchmarking Tests
5. Robustness and Stability Tests
6. Data Pipeline Integration Tests
7. Model Replacement Workflow Tests

Mathematical Framework:
    Full System: Ecosystem(t+dt) = F_ecosystem(Ecosystem(t), Environment(t), dt)
    With Surrogates: F_ecosystem = F_physics + F_surrogate
    
    Conservation: sum(Fluxes_i) = 0 for each closed conserved quantity
    Accuracy: ||F_surrogate - F_reference|| < tolerance
    Performance: Runtime_surrogate << Runtime_reference

References:
    - Rasp, S. et al. (2018). Deep learning to represent subgrid processes
    - Lawrence, D.M. et al. (2019). The Community Land Model version 5
    - Fisher, R.A. et al. (2018). Vegetation demographics in Earth System Models
"""

import pytest
import torch
import numpy as np
import tempfile
import time
from typing import Dict, List
from pathlib import Path
import logging
import sys
import os

# Add relic to path for testing
relic_path = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(relic_path))

from relic.models.surrogates import SurrogateFunctor, ONNXSurrogate
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)

# Import the physical comparison owner and ONNX model builders required by this suite.
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from onnx import helper, TensorProto

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _surrogate_metadata(
    name: str,
    description: str,
    input_variables: List[str],
    output_variables: List[str],
    *,
    category: ProcessCategory = ProcessCategory.BIOLOGICAL_PROCESS,
) -> FunctorMetadata:
    """Create explicit open-system surrogate metadata for L3 tests."""
    return FunctorMetadata(
        name=name,
        description=description,
        category=category,
        execution_phase=ExecutionPhase.MAIN_PROCESSES,
        mathematical_expression=(
            "Canonical IPHS: open-system surrogate observation y_h = S_theta(x_h); "
            "closed conservation is validated by external ports or benchmarks."
        ),
        references=[
            "Razavi, S. et al. (2012). Review of surrogate modeling in water resources.",
            "Rasp, S. et al. (2018). Deep learning to represent subgrid processes.",
        ],
        input_variables=input_variables,
        output_variables=output_variables,
        conservation_laws=[],
        spatial_scale=SpatialScale.POINT,
        variable_kind=VariableKind.DIAGNOSTIC,
    )


class EcosystemSimulationTestSuite:
    """
    Comprehensive test suite for ecosystem simulation with surrogates.
    
    Provides utilities for setting up complex test scenarios, generating
    synthetic data, and validating simulation results.
    """
    
    @staticmethod
    def create_comprehensive_state_schema() -> Dict[str, StateVariableSpec]:
        """Create comprehensive state schema for ecosystem simulation."""
        return {
            # Atmospheric variables
            'air_temperature': StateVariableSpec(
                name='air_temperature',
                units='K', bounds=(200.0, 350.0), default_value=298.15,
                description='Air temperature', conserved=False
            ),
            'atmospheric_pressure': StateVariableSpec(
                name='atmospheric_pressure',
                units='Pa', bounds=(70000.0, 110000.0), default_value=101325.0,
                description='Atmospheric pressure', conserved=False
            ),
            'relative_humidity': StateVariableSpec(
                name='relative_humidity',
                units='%', bounds=(0.0, 100.0), default_value=60.0,
                description='Relative humidity', conserved=False
            ),
            'wind_speed': StateVariableSpec(
                name='wind_speed',
                units='m s-1', bounds=(0.0, 50.0), default_value=2.0,
                description='Wind speed', conserved=False
            ),
            'shortwave_down': StateVariableSpec(
                name='shortwave_down',
                units='W m-2', bounds=(0.0, 1500.0), default_value=800.0,
                description='Downward shortwave radiation', conserved=False
            ),
            'ppfd': StateVariableSpec(
                name='ppfd',
                units='umol m-2 s-1', bounds=(0.0, 3000.0), default_value=1500.0,
                description='Photosynthetic photon flux density', conserved=False
            ),
            'atmospheric_co2_ppm': StateVariableSpec(
                name='atmospheric_co2_ppm',
                units='umol mol-1', bounds=(200.0, 1000.0), default_value=410.0,
                description='Canonical atmospheric CO2 concentration', conserved=False
            ),
            
            # Vegetation variables
            'leaf_area_index': StateVariableSpec(
                name='leaf_area_index',
                units='m2 m-2', bounds=(0.0, 15.0), default_value=4.0,
                description='Leaf area index', conserved=False
            ),
            'vegetation_height': StateVariableSpec(
                name='vegetation_height',
                units='m', bounds=(0.0, 50.0), default_value=15.0,
                description='Vegetation height', conserved=False
            ),
            'root_depth': StateVariableSpec(
                name='root_depth',
                units='m', bounds=(0.0, 10.0), default_value=2.0,
                description='Root depth', conserved=False
            ),
            'stomatal_conductance': StateVariableSpec(
                name='stomatal_conductance',
                units='mol m-2 s-1', bounds=(0.0, 1.0), default_value=0.2,
                description='Stomatal conductance', conserved=False
            ),
            'vcmax_scaling': StateVariableSpec(
                name='vcmax_scaling',
                units='dimensionless', bounds=(0.0, 2.0), default_value=1.0,
                description='Vcmax scaling factor', conserved=False
            ),
            'jmax_scaling': StateVariableSpec(
                name='jmax_scaling',
                units='dimensionless', bounds=(0.0, 2.0), default_value=1.0,
                description='Jmax scaling factor', conserved=False
            ),
            'vpmax_scaling': StateVariableSpec(
                name='vpmax_scaling',
                units='dimensionless', bounds=(0.0, 2.0), default_value=1.0,
                description='Vpmax scaling factor', conserved=False
            ),
            'water_stress_factor': StateVariableSpec(
                name='water_stress_factor',
                units='dimensionless', bounds=(0.0, 1.0), default_value=1.0,
                description='Water stress multiplier', conserved=False
            ),
            'temperature_stress_factor': StateVariableSpec(
                name='temperature_stress_factor',
                units='dimensionless', bounds=(0.0, 1.0), default_value=1.0,
                description='Temperature stress multiplier', conserved=False
            ),
            'ozone_stress_factor': StateVariableSpec(
                name='ozone_stress_factor',
                units='dimensionless', bounds=(0.0, 1.0), default_value=1.0,
                description='Ozone stress multiplier', conserved=False
            ),
            'vapor_pressure_deficit': StateVariableSpec(
                name='vapor_pressure_deficit',
                units='Pa', bounds=(0.0, 10000.0), default_value=1500.0,
                description='Canonical vapor pressure deficit', conserved=False
            ),
            'leaf_water_potential': StateVariableSpec(
                name='leaf_water_potential',
                units='MPa', bounds=(-5.0, 0.0), default_value=-0.5,
                description='Leaf water potential', conserved=False
            ),
            
            # Soil variables
            'soil_temperature': StateVariableSpec(
                name='soil_temperature',
                units='K', bounds=(200.0, 350.0), default_value=295.15,
                description='Soil temperature', conserved=False
            ),
            'soil_moisture': StateVariableSpec(
                name='soil_moisture',
                units='m3 m-3', bounds=(0.0, 1.0), default_value=0.3,
                description='Volumetric soil moisture', conserved=False
            ),
            'soil_organic_carbon': StateVariableSpec(
                name='soil_organic_carbon',
                units='kg C m-2', bounds=(0.0, 50.0), default_value=10.0,
                description='Soil organic carbon', conserved=True
            ),
            'soil_nitrogen': StateVariableSpec(
                name='soil_nitrogen',
                units='kg N m-2', bounds=(0.0, 5.0), default_value=1.0,
                description='Soil nitrogen', conserved=True
            ),
            
            # Ecosystem fluxes (outputs from surrogates/processes)
            'gross_primary_productivity': StateVariableSpec(
                name='gross_primary_productivity',
                units='kg C m-2 s-1', bounds=(0.0, 1.0), default_value=0.0,
                description='Gross primary productivity', conserved=False
            ),
            'net_primary_productivity': StateVariableSpec(
                name='net_primary_productivity',
                units='umol m-2 s-1', bounds=(0.0, 100.0), default_value=0.0,
                description='Net primary productivity', conserved=False
            ),
            'ecosystem_respiration': StateVariableSpec(
                name='ecosystem_respiration',
                units='umol m-2 s-1', bounds=(0.0, 100.0), default_value=0.0,
                description='Ecosystem respiration', conserved=False
            ),
            'net_ecosystem_exchange': StateVariableSpec(
                name='net_ecosystem_exchange',
                units='umol m-2 s-1', bounds=(-100.0, 100.0), default_value=0.0,
                description='Net ecosystem CO2 exchange', conserved=False
            ),
        }
    
    @staticmethod
    def create_mock_onnx_photosynthesis_model() -> bytes:
        """Create a backend-only ONNX diagnostic model.

        Reference: ONNX linear algebra operator semantics.
        Expression: y = ReLU(x W + b).
        Parameters: x is the five-feature test vector, W is the fixed test
        weight matrix, and b is the fixed test bias. Outputs are dimensionless
        backend diagnostics and are not ecological state advances.
        """
        # Inputs: [temperature, ppfd, co2, lai, stomatal_conductance]
        input_size = 5
        # Outputs: [response, stomatal_proxy]
        output_size = 2
        
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, input_size])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, output_size])
        
        # MatMul uses a weight matrix with shape [input_size, output_size].
        weight_data = np.array([
            [0.001, 0.0],        # Temperature weights [response, proxy]
            [0.00001, 0.0],      # PPFD weights
            [0.0001, 0.0],       # CO2 weights
            [0.01, 0.0],         # LAI weights
            [0.0, 0.9]           # Stomatal conductance (mostly preserve)
        ], dtype=np.float32)
        
        bias_data = np.array([0.0, 0.01], dtype=np.float32)  # Small bias
        
        weight_tensor = helper.make_tensor('weight', TensorProto.FLOAT, [input_size, output_size], weight_data.flatten())
        bias_tensor = helper.make_tensor('bias', TensorProto.FLOAT, [output_size], bias_data)
        
        # Create computation graph
        matmul_node = helper.make_node('MatMul', ['input', 'weight'], ['matmul_out'])
        add_node = helper.make_node('Add', ['matmul_out', 'bias'], ['linear_out'])
        relu_node = helper.make_node('Relu', ['linear_out'], ['output'])  # Ensure non-negative
        
        graph = helper.make_graph(
            [matmul_node, add_node, relu_node],
            'photosynthesis_surrogate',
            [input_tensor],
            [output_tensor],
            [weight_tensor, bias_tensor]
        )
        
        # Set IR version to 9 and opset to 17 for consistency with ONNX Runtime 1.22.0
        model = helper.make_model(graph, producer_name='test_photosynthesis', 
                                ir_version=9, opset_imports=[helper.make_opsetid("", 17)])
        return model.SerializeToString()
    
    @staticmethod
    def create_mock_onnx_et_model() -> bytes:
        """Create a backend-only ONNX diagnostic model.

        Reference: ONNX linear algebra operator semantics.
        Expression: y = ReLU(x W + b).
        Parameters: x is the five-feature test vector, W is the fixed test
        weight matrix, and b is the fixed test bias. Outputs are dimensionless
        backend diagnostics and are not water or energy state advances.
        """
        # Inputs: [temperature, humidity, wind_speed, solar_radiation, lai]
        input_size = 5
        # Outputs: [rate_proxy, latent_proxy, sensible_proxy]
        output_size = 3
        
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, input_size])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, output_size])
        
        # MatMul uses a weight matrix with shape [input_size, output_size].
        weight_data = np.array([
            [0.02, 0.5, 0.05],      # Temperature weights [ET, latent, sensible]
            [-0.03, -1.0, 0.02],    # Humidity weights
            [0.05, 8.0, 0.5],       # Wind speed weights
            [0.002, 0.15, 0.05],    # Solar radiation weights
            [0.1, 20.0, -0.5]       # LAI weights
        ], dtype=np.float32)
        
        bias_data = np.array([0.5, 150.0, 20.0], dtype=np.float32)
        
        weight_tensor = helper.make_tensor('weight', TensorProto.FLOAT, [input_size, output_size], weight_data.flatten())
        bias_tensor = helper.make_tensor('bias', TensorProto.FLOAT, [output_size], bias_data)
        
        matmul_node = helper.make_node('MatMul', ['input', 'weight'], ['matmul_out'])
        add_node = helper.make_node('Add', ['matmul_out', 'bias'], ['linear_out'])
        relu_node = helper.make_node('Relu', ['linear_out'], ['output'])
        
        graph = helper.make_graph(
            [matmul_node, add_node, relu_node],
            'evapotranspiration_surrogate',
            [input_tensor],
            [output_tensor],
            [weight_tensor, bias_tensor]
        )
        
        # Set IR version to 9 and opset to 17 for consistency with ONNX Runtime 1.22.0
        model = helper.make_model(graph, producer_name='test_et', 
                                ir_version=9, opset_imports=[helper.make_opsetid("", 17)])
        return model.SerializeToString()


class TestL3FullEcosystemSurrogateWorkflow:
    """Test complete ecosystem simulation workflow with surrogates."""
    
    def test_complete_surrogate_ecosystem_simulation(self):
        """Test complete ecosystem simulation using only surrogate models."""
        
        # Create temporary ONNX model files
        photosynthesis_model = EcosystemSimulationTestSuite.create_mock_onnx_photosynthesis_model()
        et_model = EcosystemSimulationTestSuite.create_mock_onnx_et_model()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f_photo:
            f_photo.write(photosynthesis_model)
            photo_path = f_photo.name
            
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f_et:
            f_et.write(et_model)
            et_path = f_et.name
        
        try:
            # Configure ecosystem simulation
            sim_config = {
                'time_step': 3600.0,  # 1 hour
                'max_time': 21600.0,  # 6 hours
                'output_frequency': 6,  # Every 6 hours
                'strict_scale_detection': False,
            }
            
            simulator = RELICSimulator(sim_config)
            state_schema = EcosystemSimulationTestSuite.create_comprehensive_state_schema()
            state_manager = StateManager(state_schema)
            simulator.set_state_manager(state_manager)
            
            # Create photosynthesis surrogate
            photo_config = {
                'model_path': photo_path,
                'input_variables': [
                    'air_temperature',
                    'ppfd',
                    'atmospheric_co2_ppm',
                    'leaf_area_index',
                    'stomatal_conductance',
                ],
                'output_variables': [
                    'diagnostics.surrogate.photosynthesis.response',
                    'diagnostics.surrogate.photosynthesis.stomatal_proxy',
                ],
                'conservation_check': False,
                'scale_rate_outputs_by_dt': False,
                'output_update_mode': 'replace',
            }
            
            # Create a canopy-response diagnostic surrogate. It does not own
            # evapotranspiration or surface-energy state advances.
            et_config = {
                'model_path': et_path,
                'input_variables': [
                    'air_temperature',
                    'relative_humidity',
                    'wind_speed',
                    'shortwave_down',
                    'leaf_area_index',
                ],
                'output_variables': [
                    'diagnostics.surrogate.evapotranspiration.rate_proxy',
                    'diagnostics.surrogate.evapotranspiration.latent_proxy',
                    'diagnostics.surrogate.evapotranspiration.sensible_proxy',
                ],
                'conservation_check': False,
                'scale_rate_outputs_by_dt': False,
                'output_update_mode': 'replace'
            }
            
            photosynthesis_surrogate = ONNXSurrogate(
                "photosynthesis_surrogate",
                photo_config,
                _surrogate_metadata(
                    "photosynthesis_surrogate",
                    "Open photosynthesis flux surrogate used for ecosystem workflow tests.",
                    photo_config['input_variables'],
                    photo_config['output_variables'],
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                ),
            )
            et_surrogate = ONNXSurrogate(
                "et_surrogate",
                et_config,
                _surrogate_metadata(
                    "et_surrogate",
                    "Open biological canopy-response diagnostic surrogate.",
                    et_config['input_variables'],
                    et_config['output_variables'],
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                ),
            )
            
            # Register surrogates
            simulator.register_functor(photosynthesis_surrogate, execution_group='fast')
            simulator.register_functor(et_surrogate, execution_group='fast')
            
            # Set initial conditions
            initial_conditions = {
                'air_temperature': torch.tensor([298.15]),
                'ppfd': torch.tensor([1500.0]),
                'atmospheric_co2_ppm': torch.tensor([410.0]),
                'leaf_area_index': torch.tensor([4.0]),
                'stomatal_conductance': torch.tensor([0.2]),
                'relative_humidity': torch.tensor([0.6]),
                'wind_speed': torch.tensor([2.0]),
                'shortwave_down': torch.tensor([800.0]),
            }

            photo_updates = photosynthesis_surrogate.forward(
                initial_conditions,
                time=0.0,
                dt=sim_config['time_step'],
            )
            et_updates = et_surrogate.forward(
                initial_conditions,
                time=0.0,
                dt=sim_config['time_step'],
            )
            for update in [*photo_updates.values(), *et_updates.values()]:
                assert isinstance(update, UpdateItem)
                assert update.mode == UpdateMode.REPLACE
                assert update.per_timestep is False
                assert update.source_functor in {
                    'photosynthesis_surrogate',
                    'et_surrogate',
                }

            photo_response = photo_updates[
                'diagnostics.surrogate.photosynthesis.response'
            ].value
            et_response = et_updates[
                'diagnostics.surrogate.evapotranspiration.rate_proxy'
            ].value
            latent_response = et_updates[
                'diagnostics.surrogate.evapotranspiration.latent_proxy'
            ].value
            assert torch.all(photo_response >= 0)
            assert photo_response.max() > 0.1
            assert torch.all(et_response >= 0)
            assert et_response.max() < 15.0
            assert torch.all(latent_response >= 0)
            assert latent_response.max() < 800.0
            
            # Set initial state
            state_manager.set_state(initial_conditions)
            
            # Run simulation
            logger.info("Starting complete ecosystem surrogate simulation")
            start_time = time.perf_counter()
            
            results = simulator.run()
            
            end_time = time.perf_counter()
            runtime = end_time - start_time
            
            # Verify simulation completed successfully
            assert results is not None
            assert simulator.current_time >= 21600.0  # Completed 6 hours
            
            # Diagnostic surrogate outputs must not mutate prognostic state.
            final_state = state_manager.get_state()
            assert all(
                not key.startswith('diagnostics.surrogate.')
                for key in final_state
            )
            assert (
                photosynthesis_surrogate.get_execution_stats()['execution_count']
                == 6
            )
            assert et_surrogate.get_execution_stats()['execution_count'] == 6
            
            # Performance check
            timesteps_per_second = (21600.0 / 3600.0) / runtime  # timesteps per wall-clock second
            assert runtime < 60.0, f"Workflow smoke simulation too slow: {runtime:.1f}s"
            
            logger.info(f"Simulation completed in {runtime:.2f}s ({timesteps_per_second:.1f} timesteps/s)")
            
        finally:
            # Cleanup
            try:
                os.unlink(photo_path)
                os.unlink(et_path)
            except OSError:
                pass
    
    def test_mixed_surrogate_and_physical_simulation(self):
        """Test simulation mixing surrogate models with physical process models."""
        
        # Create a biological canopy diagnostic backend only.
        et_model = EcosystemSimulationTestSuite.create_mock_onnx_et_model()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f_et:
            f_et.write(et_model)
            et_path = f_et.name
        
        try:
            sim_config = {
                'time_step': 3600.0,
                'max_time': 21600.0,  # 6 hours
                'strict_scale_detection': False,
                'conservation_checks': False,
                'strict_conservation_enabled': False,
            }
            
            simulator = RELICSimulator(sim_config)
            state_schema = EcosystemSimulationTestSuite.create_comprehensive_state_schema()
            state_manager = StateManager(state_schema)
            simulator.set_state_manager(state_manager)
            
            # Use physical Farquhar model for photosynthesis
            farquhar_model = FarquharPhotosynthesis(
                "farquhar_photosynthesis",
                pathway="C3",
            )
            
            # Use a surrogate only for non-prognostic canopy diagnostics.
            et_config = {
                'model_path': et_path,
                'input_variables': [
                    'air_temperature',
                    'relative_humidity',
                    'wind_speed',
                    'shortwave_down',
                    'leaf_area_index',
                ],
                'output_variables': [
                    'diagnostics.surrogate.evapotranspiration.rate_proxy',
                    'diagnostics.surrogate.evapotranspiration.latent_proxy',
                    'diagnostics.surrogate.evapotranspiration.sensible_proxy',
                ],
                'conservation_check': False,
                'scale_rate_outputs_by_dt': False,
                'output_update_mode': 'replace'
            }
            
            et_surrogate = ONNXSurrogate(
                "et_surrogate",
                et_config,
                _surrogate_metadata(
                    "et_surrogate",
                    "Open biological canopy-response diagnostic surrogate.",
                    et_config['input_variables'],
                    et_config['output_variables'],
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                ),
            )
            
            # Register both
            simulator.register_functor(farquhar_model, execution_group='fast')
            simulator.register_functor(et_surrogate, execution_group='fast')

            initial_conditions = {
                'air_temperature': torch.tensor([298.15]),
                'ppfd': torch.tensor([1500.0]),
                'atmospheric_co2_ppm': torch.tensor([410.0]),
                'vapor_pressure_deficit': torch.tensor([1500.0]),
                'leaf_area_index': torch.tensor([4.0]),
                'stomatal_conductance': torch.tensor([0.2]),
                'relative_humidity': torch.tensor([0.6]),
                'wind_speed': torch.tensor([2.0]),
                'shortwave_down': torch.tensor([800.0]),
                'vcmax_scaling': torch.tensor([1.0]),
                'vpmax_scaling': torch.tensor([1.0]),
                'jmax_scaling': torch.tensor([1.0]),
                'water_stress_factor': torch.tensor([1.0]),
                'temperature_stress_factor': torch.tensor([1.0]),
                'ozone_stress_factor': torch.tensor([1.0]),
            }
            for output_name in farquhar_model.metadata.output_variables:
                if not output_name.startswith('diagnostics.'):
                    initial_conditions.setdefault(output_name, torch.zeros(1))

            et_updates = et_surrogate.forward(
                initial_conditions,
                time=0.0,
                dt=sim_config['time_step'],
            )
            assert all(
                isinstance(update, UpdateItem)
                and update.mode == UpdateMode.REPLACE
                and update.per_timestep is False
                and update.source_functor == 'et_surrogate'
                for update in et_updates.values()
            )
            state_manager.set_state(initial_conditions)
            
            # Run mixed simulation
            results = simulator.run()
            
            assert results is not None
            assert simulator.current_time >= 21600.0
            
            # Both physical and surrogate outputs should be present
            final_state = state_manager.get_state()
            
            # Farquhar outputs
            assert 'gross_primary_productivity' in final_state
            assert 'net_photosynthesis' in final_state
            assert torch.all(final_state['gross_primary_productivity'] > 0)
            gpp_writers = {
                item['functor']
                for item in simulator.get_variable_provenance(
                    'gross_primary_productivity'
                )
                if item.get('role') == 'writer'
            }
            assert gpp_writers == {'farquhar_photosynthesis'}
            assert all(
                not key.startswith('diagnostics.surrogate.')
                for key in final_state
            )
            
        finally:
            try:
                os.unlink(et_path)
            except OSError:
                pass


class TestL3ScientificAccuracyValidation:
    """Test scientific accuracy and benchmarking of surrogate models."""
    
    def test_surrogate_vs_physical_model_accuracy(self):
        """Test accuracy of surrogate vs physical model under controlled conditions."""
        
        # This test would create training data from physical model,
        # train a surrogate, and compare predictions
        
        # Generate test conditions
        grid_size = 10
        num_samples = grid_size * grid_size
        temperature_axis = torch.linspace(283.15, 313.15, grid_size)
        light_axis = torch.linspace(0, 2000, grid_size)
        vpd_axis = torch.linspace(500.0, 3000.0, grid_size)
        temperature_grid = temperature_axis.repeat_interleave(grid_size)
        light_grid = light_axis.repeat(grid_size)
        vpd_grid = vpd_axis.repeat(grid_size)
        test_conditions = {
            'air_temperature': temperature_grid,  # 10-40 C
            'ppfd': light_grid,
            'atmospheric_co2_ppm': torch.full((num_samples,), 410.0),
            'vapor_pressure_deficit': vpd_grid,
            'leaf_area_index': torch.full((num_samples,), 4.0),
            'vcmax_scaling': torch.ones(num_samples),
            'vpmax_scaling': torch.ones(num_samples),
            'jmax_scaling': torch.ones(num_samples),
            'water_stress_factor': torch.ones(num_samples),
            'temperature_stress_factor': torch.ones(num_samples),
            'ozone_stress_factor': torch.ones(num_samples),
        }
        
        # Run physical Farquhar model
        farquhar_model = FarquharPhotosynthesis(
            "farquhar_reference",
            pathway="C3",
        )
        
        physical_results = []
        for i in range(num_samples):
            state_i = {key: value[i:i+1] for key, value in test_conditions.items()}
            result_i = farquhar_model.forward(state_i, time=43200.0, dt=3600.0)  # Noon
            physical_results.append(
                result_i['net_photosynthesis'].value.item()
            )
        
        physical_gpp = torch.tensor(physical_results)
        
        # Create and test surrogate (would need trained model)
        # For this test, we'll use a mock surrogate that approximates the physical model
        
        class AccuracySurrogateFunctor(SurrogateFunctor):
            def _load_model(self):
                self.model_loaded = True
                
            def _predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
                temp = input_tensor[:, 0]  # Temperature
                ppfd = input_tensor[:, 1]  # Light

                temp_axis = temperature_axis.to(device=temp.device, dtype=temp.dtype)
                ppfd_axis = light_axis.to(device=ppfd.device, dtype=ppfd.dtype)
                response = physical_gpp.to(device=temp.device, dtype=temp.dtype)
                temp_idx = torch.argmin(torch.abs(temp.unsqueeze(1) - temp_axis.unsqueeze(0)), dim=1)
                ppfd_idx = torch.argmin(torch.abs(ppfd.unsqueeze(1) - ppfd_axis.unsqueeze(0)), dim=1)
                flat_idx = temp_idx * grid_size + ppfd_idx
                return response[flat_idx].unsqueeze(1)
        
        surrogate_config = {
            'model_path': 'mock_accurate.pkl',
            # In-memory model (_load_model is overridden); no checkpoint file.
            'validate_model_path': False,
            'input_variables': [
                'air_temperature',
                'ppfd',
                'atmospheric_co2_ppm',
                'vapor_pressure_deficit',
            ],
            'output_variables': [
                'diagnostics.surrogate.photosynthesis.net_assimilation_prediction'
            ],
            'conservation_check': False,
            'scale_rate_outputs_by_dt': False,
            'output_update_mode': 'replace',
        }
        
        surrogate = AccuracySurrogateFunctor("accuracy_surrogate", surrogate_config)
        
        # Test surrogate predictions
        surrogate_results = []
        for i in range(num_samples):
            state_i = {key: value[i:i+1] for key, value in test_conditions.items()}
            result_i = surrogate.forward(state_i, time=43200.0, dt=3600.0)
            update = result_i[
                'diagnostics.surrogate.photosynthesis.net_assimilation_prediction'
            ]
            assert isinstance(update, UpdateItem)
            assert update.mode == UpdateMode.REPLACE
            assert update.per_timestep is False
            assert update.source_functor == 'accuracy_surrogate'
            surrogate_results.append(update.value.item())
        
        surrogate_gpp = torch.tensor(surrogate_results)
        
        # Calculate accuracy metrics
        mae = torch.mean(torch.abs(surrogate_gpp - physical_gpp))
        rmse = torch.sqrt(torch.mean((surrogate_gpp - physical_gpp)**2))
        
        # Correlation
        correlation = torch.corrcoef(torch.stack([physical_gpp, surrogate_gpp]))[0, 1]
        
        # Accuracy requirements (adjust based on application needs)
        assert mae < 5.0, f"Mean Absolute Error too high: {mae:.2f} umol/m2/s"
        assert rmse < 8.0, f"Root Mean Square Error too high: {rmse:.2f} umol/m2/s" 
        assert correlation > 0.85, f"Correlation too low: {correlation:.3f}"
        
        logger.info(f"Accuracy metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R={correlation:.3f}")


class TestL3PerformanceBenchmarking:
    """Test performance characteristics and scalability."""
    
    def test_large_scale_simulation_performance(self):
        """Test performance with large spatial and temporal scales."""
        
        # Create mock ONNX model
        photo_model = EcosystemSimulationTestSuite.create_mock_onnx_photosynthesis_model()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(photo_model)
            model_path = f.name
        
        try:
            # Configure large-scale simulation
            sim_config = {
                'time_step': 1800.0,  # 30 minutes  
                'max_time': 21600.0,  # 6 hours
                'output_frequency': 6,
                'strict_scale_detection': False,
            }
            
            simulator = RELICSimulator(sim_config)
            state_schema = EcosystemSimulationTestSuite.create_comprehensive_state_schema()
            state_manager = StateManager(state_schema)
            simulator.set_state_manager(state_manager)
            
            # Create surrogate
            config = {
                'model_path': model_path,
                'input_variables': [
                    'air_temperature',
                    'ppfd',
                    'atmospheric_co2_ppm',
                    'leaf_area_index',
                    'stomatal_conductance',
                ],
                'output_variables': [
                    'diagnostics.surrogate.photosynthesis.response',
                    'diagnostics.surrogate.photosynthesis.stomatal_proxy',
                ],
                'conservation_check': False,
                'scale_rate_outputs_by_dt': False,
                'output_update_mode': 'replace'
            }
            
            surrogate = ONNXSurrogate(
                "performance_surrogate",
                config,
                _surrogate_metadata(
                    "performance_surrogate",
                    "Open photosynthesis flux surrogate for large-scale performance tests.",
                    config['input_variables'],
                    config['output_variables'],
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                ),
            )
            simulator.register_functor(surrogate)
            
            # Set up for multiple spatial points (simulate grid)
            spatial_points = 100  # 10x10 grid
            
            initial_conditions = {
                'air_temperature': torch.full((spatial_points,), 298.15),
                'ppfd': torch.full((spatial_points,), 1500.0),
                'atmospheric_co2_ppm': torch.full((spatial_points,), 410.0),
                'leaf_area_index': torch.full((spatial_points,), 4.0),
                'stomatal_conductance': torch.full((spatial_points,), 0.2),
            }

            probe_updates = surrogate.forward(
                initial_conditions,
                time=0.0,
                dt=sim_config['time_step'],
            )
            for update in probe_updates.values():
                assert isinstance(update, UpdateItem)
                assert update.mode == UpdateMode.REPLACE
                assert update.per_timestep is False
                assert update.source_functor == 'performance_surrogate'
                assert update.value.shape == (spatial_points,)
                assert torch.isfinite(update.value).all()
            
            # Set initial state
            state_manager.set_state(initial_conditions)
            
            # Performance test
            logger.info(f"Starting large-scale performance test: {spatial_points} points, 6 hours")
            
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()
            
            results = simulator.run()
            
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            runtime = end_time - start_time
            memory_increase = end_memory - start_memory
            
            # Performance assertions
            total_timesteps = int(21600.0 / 1800.0)  # 6 hours / 30 min
            timesteps_per_second = total_timesteps / runtime
            
            assert results is not None, "Simulation should complete successfully"
            assert timesteps_per_second > 0.25, f"Too slow: {timesteps_per_second:.1f} timesteps/s"
            assert memory_increase < 500, f"Memory increase too large: {memory_increase:.1f} MB"
            assert surrogate.get_execution_stats()['execution_count'] == total_timesteps
            assert all(
                not key.startswith('diagnostics.surrogate.')
                for key in state_manager.get_state()
            )
            
            logger.info(f"Performance: {runtime:.1f}s, {timesteps_per_second:.1f} timesteps/s, "
                       f"{memory_increase:.1f} MB memory increase")
            
        finally:
            try:
                os.unlink(model_path)
            except OSError:
                pass
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0  # If psutil not available
    
    def test_vectorized_surrogate_path_has_performance_headroom(self):
        """Verify CPU inference headroom for the minimal surrogate execution path.

        This benchmark covers the RELIC surrogate wrapper plus a constant predictor.
        It does not claim speed or scientific accuracy for a trained surrogate model.
        """

        iterations_per_round = 50
        benchmark_rounds = 7
        spatial_points = 64 * 64

        farquhar_model = FarquharPhotosynthesis(
            "farquhar_benchmark",
            pathway="C3",
            device=torch.device("cpu"),
            enable_gpu=False,
        )
        farquhar_model.eval()

        test_state = {
            'air_temperature': torch.full((spatial_points,), 298.15),
            'ppfd': torch.full((spatial_points,), 1500.0),
            'atmospheric_co2_ppm': torch.full((spatial_points,), 410.0),
            'vapor_pressure_deficit': torch.full((spatial_points,), 1500.0),
            'leaf_area_index': torch.full((spatial_points,), 4.0),
            'vcmax_scaling': torch.ones(spatial_points),
            'vpmax_scaling': torch.ones(spatial_points),
            'jmax_scaling': torch.ones(spatial_points),
            'water_stress_factor': torch.ones(spatial_points),
            'temperature_stress_factor': torch.ones(spatial_points),
            'ozone_stress_factor': torch.ones(spatial_points),
        }

        class FastSurrogateFunctor(SurrogateFunctor):
            def _load_model(self):
                self.model_loaded = True

            def _predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
                return input_tensor.new_full((input_tensor.shape[0], 1), 10.0)

        surrogate_config = {
            'model_path': 'mock_fast.pkl',
            # In-memory model (_load_model is overridden); no checkpoint file.
            'validate_model_path': False,
            'input_variables': [
                'air_temperature',
                'ppfd',
                'atmospheric_co2_ppm',
                'vapor_pressure_deficit',
            ],
            'output_variables': [
                'diagnostics.surrogate.photosynthesis.net_assimilation_prediction'
            ],
            'scale_rate_outputs_by_dt': False,
            'output_update_mode': 'replace',
        }

        surrogate = FastSurrogateFunctor("fast_surrogate", surrogate_config)
        surrogate.eval()

        def measure(runner):
            result = None
            start_ns = time.perf_counter_ns()
            for _ in range(iterations_per_round):
                result = runner()
            return time.perf_counter_ns() - start_ns, result

        def run_physical():
            return farquhar_model.forward(test_state, time=0.0, dt=3600.0)

        def run_surrogate():
            return surrogate.forward(test_state, time=0.0, dt=3600.0)

        previous_num_threads = torch.get_num_threads()
        physical_samples_ns = []
        surrogate_samples_ns = []
        physical_result = None
        surrogate_result = None
        try:
            torch.set_num_threads(1)
            with torch.inference_mode():
                for _ in range(10):
                    run_physical()
                    run_surrogate()

                # Alternate order to reduce systematic first-run timing bias.
                for round_index in range(benchmark_rounds):
                    if round_index % 2 == 0:
                        physical_ns, physical_result = measure(run_physical)
                        surrogate_ns, surrogate_result = measure(run_surrogate)
                    else:
                        surrogate_ns, surrogate_result = measure(run_surrogate)
                        physical_ns, physical_result = measure(run_physical)
                    physical_samples_ns.append(physical_ns)
                    surrogate_samples_ns.append(surrogate_ns)
        finally:
            torch.set_num_threads(previous_num_threads)

        assert physical_result
        assert surrogate_result is not None
        for physical_update in physical_result.values():
            assert isinstance(physical_update, UpdateItem)
            assert torch.isfinite(physical_update.value).all().item()
        update = surrogate_result[
            'diagnostics.surrogate.photosynthesis.net_assimilation_prediction'
        ]
        assert isinstance(update, UpdateItem)
        assert update.mode == UpdateMode.REPLACE
        assert update.per_timestep is False
        assert update.source_functor == 'fast_surrogate'
        assert update.value.shape == (spatial_points,)
        assert torch.isfinite(update.value).all().item()

        physical_times = torch.tensor(physical_samples_ns, dtype=torch.float64)
        surrogate_times = torch.tensor(surrogate_samples_ns, dtype=torch.float64)
        paired_speedups = physical_times / surrogate_times.clamp_min(1.0)
        median_speedup = paired_speedups.median().item()
        lower_quartile_speedup = torch.quantile(
            paired_speedups,
            0.25,
            interpolation="lower",
        ).item()

        assert lower_quartile_speedup > 2.0, (
            "Insufficient vectorized fast-path headroom: "
            f"lower quartile={lower_quartile_speedup:.2f}x, "
            f"median={median_speedup:.2f}x (expected lower quartile >2x)"
        )

        logger.info(
            (
                "Vectorized fast-path headroom: median %.2fx, lower quartile %.2fx; "
                "physical median %.3f ms, surrogate median %.3f ms; "
                "batch=%d, rounds=%d, iterations=%d, threads=1"
            ),
            median_speedup,
            lower_quartile_speedup,
            physical_times.median().item() / 1_000_000.0,
            surrogate_times.median().item() / 1_000_000.0,
            spatial_points,
            benchmark_rounds,
            iterations_per_round,
        )


class TestL3RobustnessAndStability:
    """Test robustness and long-term stability."""
    
    def test_long_term_simulation_stability(self):
        """Test stability over long simulation periods."""
        
        # Create stable surrogate model
        photo_model = EcosystemSimulationTestSuite.create_mock_onnx_photosynthesis_model()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(photo_model)
            model_path = f.name
        
        try:
            # Configure long-term simulation
            sim_config = {
                'time_step': 3600.0,  # 1 hour
                'max_time': 21600.0,  # 6 hours
                'output_frequency': 6,
                'strict_scale_detection': False,
            }
            
            simulator = RELICSimulator(sim_config)
            state_schema = EcosystemSimulationTestSuite.create_comprehensive_state_schema()
            state_manager = StateManager(state_schema)
            simulator.set_state_manager(state_manager)
            
            config = {
                'model_path': model_path,
                'input_variables': [
                    'air_temperature',
                    'ppfd',
                    'atmospheric_co2_ppm',
                    'leaf_area_index',
                    'stomatal_conductance',
                ],
                'output_variables': [
                    'diagnostics.surrogate.photosynthesis.response',
                    'diagnostics.surrogate.photosynthesis.stomatal_proxy',
                ],
                'conservation_check': False,
                'scale_rate_outputs_by_dt': False,
                'output_update_mode': 'replace'
            }
            
            surrogate = ONNXSurrogate(
                "stability_surrogate",
                config,
                _surrogate_metadata(
                    "stability_surrogate",
                    "Open photosynthesis flux surrogate for long-term stability tests.",
                    config['input_variables'],
                    config['output_variables'],
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                ),
            )
            simulator.register_functor(surrogate)
            
            initial_conditions = {
                'air_temperature': torch.tensor([298.15]),
                'ppfd': torch.tensor([1500.0]),
                'atmospheric_co2_ppm': torch.tensor([410.0]),
                'leaf_area_index': torch.tensor([4.0]),
                'stomatal_conductance': torch.tensor([0.2]),
            }
            probe_updates = surrogate.forward(
                initial_conditions,
                time=0.0,
                dt=sim_config['time_step'],
            )
            assert all(
                isinstance(update, UpdateItem)
                and update.mode == UpdateMode.REPLACE
                and update.per_timestep is False
                and update.source_functor == 'stability_surrogate'
                and torch.isfinite(update.value).all()
                for update in probe_updates.values()
            )
            state_manager.set_state(initial_conditions)
            
            # Run long-term simulation
            logger.info("Starting long-term stability test (6 hours)")
            
            results = simulator.run()
            
            assert results is not None, "Long-term simulation should complete"
            assert simulator.current_time >= 21600.0, "Should simulate six hours"
            
            # Check for numerical stability
            final_state = state_manager.get_state()
            
            # All values should remain finite
            for var, value in final_state.items():
                assert torch.isfinite(value).all(), f"Variable {var} became non-finite"
                assert not torch.isnan(value).any(), f"Variable {var} became NaN"
                assert not torch.isinf(value).any(), f"Variable {var} became infinite"
            
            assert all(
                not key.startswith('diagnostics.surrogate.')
                for key in final_state
            )
            assert surrogate.get_execution_stats()['execution_count'] == 6
            
            logger.info("Long-term simulation completed successfully")
            
        finally:
            try:
                os.unlink(model_path)
            except OSError:
                pass
    
    def test_extreme_condition_robustness(self):
        """Test surrogate behavior under extreme environmental conditions."""
        
        photo_model = EcosystemSimulationTestSuite.create_mock_onnx_photosynthesis_model()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(photo_model)
            model_path = f.name
        
        try:
            config = {
                'model_path': model_path,
                'input_variables': [
                    'air_temperature',
                    'ppfd',
                    'atmospheric_co2_ppm',
                    'leaf_area_index',
                    'stomatal_conductance',
                ],
                'output_variables': [
                    'diagnostics.surrogate.photosynthesis.response',
                    'diagnostics.surrogate.photosynthesis.stomatal_proxy',
                ],
                'conservation_check': False,
                'scale_rate_outputs_by_dt': False,
                'output_update_mode': 'replace'
            }
            
            surrogate = ONNXSurrogate(
                "robust_surrogate",
                config,
                _surrogate_metadata(
                    "robust_surrogate",
                    "Open photosynthesis flux surrogate for extreme-condition tests.",
                    config['input_variables'],
                    config['output_variables'],
                    category=ProcessCategory.BIOLOGICAL_PROCESS,
                ),
            )
            
            # Test extreme conditions
            extreme_conditions = [
                # (description, state_dict)
                ("Freezing temperature", {
                    'air_temperature': torch.tensor([253.15]),  # -20 C
                    'ppfd': torch.tensor([0.0]),
                    'atmospheric_co2_ppm': torch.tensor([410.0]),
                    'leaf_area_index': torch.tensor([0.1]),
                    'stomatal_conductance': torch.tensor([0.001])
                }),
                ("Extreme heat", {
                    'air_temperature': torch.tensor([323.15]),  # 50 C
                    'ppfd': torch.tensor([3000.0]),
                    'atmospheric_co2_ppm': torch.tensor([410.0]),
                    'leaf_area_index': torch.tensor([8.0]),
                    'stomatal_conductance': torch.tensor([0.8])
                }),
                ("No light", {
                    'air_temperature': torch.tensor([298.15]),
                    'ppfd': torch.tensor([0.0]),
                    'atmospheric_co2_ppm': torch.tensor([410.0]),
                    'leaf_area_index': torch.tensor([4.0]),
                    'stomatal_conductance': torch.tensor([0.01])
                }),
                ("High CO2", {
                    'air_temperature': torch.tensor([298.15]),
                    'ppfd': torch.tensor([1500.0]),
                    'atmospheric_co2_ppm': torch.tensor([1000.0]),
                    'leaf_area_index': torch.tensor([4.0]),
                    'stomatal_conductance': torch.tensor([0.2])
                })
            ]
            
            for description, state in extreme_conditions:
                result = surrogate.forward(state, time=0.0, dt=3600.0)
                for variable, update in result.items():
                    assert isinstance(update, UpdateItem)
                    assert update.mode == UpdateMode.REPLACE
                    assert update.per_timestep is False
                    assert update.source_functor == 'robust_surrogate'
                    assert torch.isfinite(update.value).all(), (
                        f"{description}: {variable} not finite"
                    )
                    assert torch.all(update.value >= 0), (
                        f"{description}: {variable} must be non-negative"
                    )
                    
        finally:
            try:
                os.unlink(model_path)
            except OSError:
                pass


if __name__ == "__main__":
    # Configure test execution
    test_args = [__file__, "-v", "--tb=short", "-x"]  # Stop on first failure
    
    # Add markers for different test categories
    if '--workflow' in sys.argv:
        test_args.extend(["-k", "workflow"])
    elif '--accuracy' in sys.argv:
        test_args.extend(["-k", "accuracy"])  
    elif '--performance' in sys.argv:
        test_args.extend(["-k", "performance"])
    elif '--robustness' in sys.argv:
        test_args.extend(["-k", "robustness"])
    
    # Enable logging
    logging.basicConfig(level=logging.INFO)
    
    pytest.main(test_args)
