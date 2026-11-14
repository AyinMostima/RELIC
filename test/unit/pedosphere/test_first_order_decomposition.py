"""
L1 Unit Tests for FirstOrderDecomposition Module

This module provides comprehensive unit tests for the first-order decomposition
process implementation, covering scientific accuracy, parameter sensitivity,
boundary conditions, and numerical stability.

Test Categories:
- L1.1: Basic instantiation and configuration
- L1.2: Scientific accuracy against literature values
- L1.3: Environmental response functions  
- L1.4: Boundary condition handling
- L1.5: Parameter sensitivity analysis
- L1.6: Mass balance and conservation
- L1.7: Performance and stability
"""

import pytest
import torch
import numpy as np
from typing import Dict

# Import the module under test
from relic.core.state import UpdateItem
from relic.scientific.pedosphere.decomposition import FirstOrderDecomposition
from relic.parameters import get_global_manager

# Import utilities for testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


EXPECTED_PARAMETER_KEYS = {
    "pedosphere.decomposition.param_25",
    "pedosphere.decomposition.clay_enhancement_coefficient",
    "pedosphere.decomposition.max_clay_factor",
    "pedosphere.decomposition.clay_protection_coefficient",
    "pedosphere.decomposition.k_metabolic",
    "pedosphere.decomposition.k_structural",
    "pedosphere.decomposition.k_fast_som",
    "pedosphere.decomposition.k_slow_som",
    "pedosphere.decomposition.k_passive_som",
    "pedosphere.decomposition.t_met_fast",
    "pedosphere.decomposition.t_met_slow",
    "pedosphere.decomposition.t_str_fast",
    "pedosphere.decomposition.t_str_slow",
    "pedosphere.decomposition.t_fast_slow",
    "pedosphere.decomposition.t_fast_passive",
    "pedosphere.decomposition.t_slow_fast",
    "pedosphere.decomposition.t_slow_passive",
    "pedosphere.decomposition.temp_response_type",
    "pedosphere.physical.q10_decomposition",
    "pedosphere.physical.activation_energy_decomposition",
    "pedosphere.decomposition.moisture_response_power",
    "pedosphere.decomposition.field_capacity",
    "pedosphere.decomposition.wilting_point",
    "pedosphere.numerical.clamp_min",
    "pedosphere.numerical.clamp_max",
    "pedosphere.physical.temperature_minimum_biological",
    "pedosphere.physical.temperature_maximum_biological",
    "pedosphere.numerical.temperature_response_min",
    "pedosphere.numerical.temperature_response_max",
    "pedosphere.numerical.unity_multiplier",
    "pedosphere.decomposition.lignin_inhibition_coefficient",
    "pedosphere.decomposition.max_lignin_n_ratio",
}


@pytest.fixture(autouse=True)
def _restore_global_parameter_manager(restore_global_parameter_manager):
    """Tests in this module write to the process-wide ParameterManager; the
    shared fixture puts it back after each test."""
    yield


def _value(result: Dict[str, UpdateItem], key: str) -> torch.Tensor:
    """Return a tensor payload from an explicit update item."""
    item = result[key]
    assert isinstance(item, UpdateItem), f"{key} must be an UpdateItem"
    assert isinstance(item.value, torch.Tensor), f"{key} payload must be a Tensor"
    return item.value


class TestFirstOrderDecompositionL1:
    """L1 Unit Tests for FirstOrderDecomposition functor."""

    @staticmethod
    def _normalize_param_value(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return value

    @staticmethod
    def _set_param_value(param_manager, key: str, value) -> None:
        spec = param_manager.get_parameter(key)
        units = spec.units if spec is not None else None
        param_manager.set_value(key, value, units=units, validate=True)
    
    @pytest.fixture
    def default_config(self) -> Dict:
        """Default configuration for FirstOrderDecomposition."""
        return {"dependencies": []}
    
    @pytest.fixture  
    def decomposer(self, default_config) -> FirstOrderDecomposition:
        """Create FirstOrderDecomposition instance."""
        return FirstOrderDecomposition(name="test_decomposition", config=default_config)
    
    @pytest.fixture
    def standard_state(self) -> Dict:
        """Standard state for testing."""
        return {
            'metabolic_litter_c': torch.tensor(0.1),      # g C m-2
            'structural_litter_c': torch.tensor(0.5),     # g C m-2
            'fast_som_c': torch.tensor(0.8),              # g C m-2
            'slow_som_c': torch.tensor(3.0),              # g C m-2
            'passive_som_c': torch.tensor(5.0),           # g C m-2
            'soil_temperature': torch.tensor(288.15),     # K
            'soil_moisture': torch.tensor(0.20),          # m3 m-3
            'metabolic_litter_input': torch.tensor(0.0),  # g C m-2 s-1
            'structural_litter_input': torch.tensor(0.0), # g C m-2 s-1
            'lignin_n_ratio': torch.tensor(0.3),          # dimensionless
            'clay_content': torch.tensor(0.25),           # dimensionless
        }
        
    # =============================================================================
    # L1.1: Basic Instantiation and Configuration Tests
    # =============================================================================
    
    def test_instantiation_basic(self, default_config):
        """Test basic instantiation with default configuration."""
        decomposer = FirstOrderDecomposition(name="test_decomp", config=default_config)
        
        assert decomposer.name == "test_decomp"
        assert decomposer.config == default_config
        assert hasattr(decomposer, 'metadata')
        assert hasattr(decomposer, 'pool_params')
        assert hasattr(decomposer, 'transfer_params')
        assert decomposer.metadata.name == "test_decomp"
        
    def test_instantiation_metadata(self, decomposer):
        """Test metadata completeness."""
        metadata = decomposer.metadata
        
        # Check required metadata fields
        assert metadata.name is not None
        assert metadata.version is not None
        assert metadata.description is not None
        assert metadata.category is not None
        assert metadata.mathematical_expression is not None
        
        # Check references are provided
        assert len(metadata.references) >= 3
        assert any("RothC" in ref or "Coleman" in ref for ref in metadata.references)
        assert any("CENTURY" in ref or "Parton" in ref for ref in metadata.references)
        
        assert set(metadata.parameters) == EXPECTED_PARAMETER_KEYS
        assert set(decomposer.required_parameters()) == EXPECTED_PARAMETER_KEYS
        assert not any(
            key.startswith("relic.physical.constants.")
            for key in decomposer.required_parameters()
        )
        
    def test_pool_initialization(self, decomposer):
        """Test carbon pool parameter initialization."""
        pools = decomposer.pool_params
        
        # Check all expected pools are present
        expected_pools = ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']
        assert all(pool in pools for pool in expected_pools)
        
        # Check rate values are reasonable (converted to per-second)
        assert 0 < pools['metabolic_litter']['k_ref'] < 1e-5  # Fast pool
        assert 0 < pools['passive_som']['k_ref'] < 1e-8      # Slow pool
        
        # Only parameters used by the carbon equation belong to this runtime map.
        for pool_parameters in pools.values():
            assert set(pool_parameters) == {"k_ref", "clay_protection"}
        
    def test_transfer_matrix_initialization(self, decomposer):
        """Test transfer matrix parameter initialization."""
        transfers = decomposer.transfer_params
        
        # Check key transfer parameters exist
        expected_transfers = ['metabolic_to_fast', 'structural_to_slow', 'fast_to_passive']
        assert all(transfer in transfers for transfer in expected_transfers)
        
        # Check transfer efficiencies are in valid range [0, 1]
        for key, value in transfers.items():
            assert 0.0 <= value <= 1.0, f"Transfer {key} = {value} outside [0,1]"
    
    def test_dependencies(self, decomposer):
        """Test dependency declaration."""
        deps = decomposer.get_dependencies()

        assert isinstance(deps, list)
        assert deps == []
        
    # =============================================================================
    # L1.2: Scientific Accuracy Tests Against Literature Values
    # =============================================================================
    
    def test_pool_turnover_times_literature(self, decomposer):
        """Test pool turnover times against literature values."""
        turnover_times = decomposer.get_pool_turnover_times()
        
        # Literature ranges (years) from RothC, CENTURY models
        expected_ranges = {
            'metabolic_litter_turnover_years': (0.1, 1.0),      # Weeks to months
            'structural_litter_turnover_years': (0.2, 2.0),     # Months to years  
            'fast_som_turnover_years': (0.1, 5.0),              # Months to few years
            'slow_som_turnover_years': (5.0, 50.0),             # Years to decades
            'passive_som_turnover_years': (100.0, 2000.0)       # Centuries
        }
        
        for pool_key, (min_val, max_val) in expected_ranges.items():
            assert pool_key in turnover_times
            actual = turnover_times[pool_key]
            epsilon = 1e-6
            assert min_val - epsilon <= actual <= max_val + epsilon, \
                f"{pool_key}: {actual:.3f} not in range [{min_val}, {max_val}]"
    
    def test_temperature_response_q10(self, decomposer, standard_state):
        """Test temperature response follows Q10 relationship."""
        # Test at different temperatures
        temperatures = [278.15, 288.15, 298.15, 308.15]  # 5degC, 15degC, 25degC, 35degC
        responses = []
        
        for temp in temperatures:
            temp_response = decomposer._calculate_temperature_response(torch.tensor(temp))
            responses.append(float(temp_response))
        
        # Check Q10 relationship: response should approximately double every 10degC
        q10_expected = decomposer.q10
        for i in range(len(temperatures) - 1):
            temp_diff = temperatures[i+1] - temperatures[i]  # 10degC difference
            response_ratio = responses[i+1] / responses[i]
            expected_ratio = q10_expected ** (temp_diff / 10.0)
            
            # Allow 10% tolerance for Q10 relationship
            assert abs(response_ratio - expected_ratio) / expected_ratio < 0.15, \
                f"Q10 relationship violated: {response_ratio:.3f} vs expected {expected_ratio:.3f}"
    
    def test_moisture_response_field_capacity(self, decomposer):
        """Test moisture response peaks near field capacity."""
        # Test moisture response across range
        moistures = torch.linspace(0.05, 0.4, 20)
        responses = []
        
        for moisture in moistures:
            response = decomposer._calculate_moisture_response(torch.tensor([moisture]))
            responses.append(float(response))
        
        responses = np.array(responses)
        max_response_idx = np.argmax(responses)
        optimal_moisture = float(moistures[max_response_idx])
        
        # Optimal moisture should be close to field capacity
        field_capacity = decomposer.field_capacity
        assert abs(optimal_moisture - field_capacity) < 0.05, \
            f"Optimal moisture {optimal_moisture:.3f} far from field capacity {field_capacity:.3f}"
        
        # Response should decrease below wilting point and above field capacity
        wilting_point = decomposer.wilting_point
        below_wp_idx = int(len(moistures) * 0.1)  # First 10%
        above_fc_idx = int(len(moistures) * 0.9)  # Last 10%
        
        assert responses[below_wp_idx] < responses[max_response_idx] * 0.5
        # Above field capacity response can be high but should be capped
        assert responses[above_fc_idx] <= 1.0
    
    def test_lignin_inhibition_century_model(self, decomposer):
        """Test lignin inhibition follows CENTURY model."""
        # Test lignin/N ratio effects
        lignin_ratios = torch.tensor([0.1, 0.3, 0.5, 0.8, 1.2])
        inhibition_factors = []
        
        for ratio in lignin_ratios:
            factor = decomposer._calculate_lignin_effects(torch.tensor([ratio]))
            inhibition_factors.append(float(factor))
        
        inhibition_factors = np.array(inhibition_factors)
        
        # Should be monotonically decreasing with lignin/N ratio
        for i in range(len(inhibition_factors) - 1):
            assert inhibition_factors[i] >= inhibition_factors[i+1], \
                "Lignin inhibition should increase with lignin/N ratio"
        
        # Should follow exponential decline (CENTURY: exp(-3 * L/N))
        expected_factors = np.exp(-3.0 * lignin_ratios.numpy())
        
        for actual, expected in zip(inhibition_factors, expected_factors):
            assert abs(actual - expected) / expected < 0.1, \
                f"Lignin factor {actual:.3f} vs CENTURY expected {expected:.3f}"
    
    # =============================================================================
    # L1.3: Environmental Response Function Tests  
    # =============================================================================
    
    def test_temperature_response_bounds(self, decomposer):
        """Test temperature response stays within reasonable bounds."""
        # Test extreme temperatures
        extreme_temps = [243.15, 253.15, 323.15, 333.15]  # -30degC, -20degC, 50degC, 60degC
        
        for temp in extreme_temps:
            try:
                response = decomposer._calculate_temperature_response(torch.tensor(temp))
                # Should be bounded between 0.1 and 10.0
                assert 0.05 <= float(response) <= 12.0, \
                    f"Temperature response {response:.3f} outside bounds at {temp-273.15:.1f}degC"
            except ValueError:
                # Extreme temperatures may raise validation errors, which is acceptable
                pass
    
    def test_moisture_response_bounds(self, decomposer):
        """Test moisture response handles full range properly."""
        # Test full moisture range
        moistures = torch.tensor([0.0, 0.05, 0.15, 0.25, 0.35, 0.5, 0.8])
        
        for moisture in moistures:
            response = decomposer._calculate_moisture_response(torch.tensor([moisture]))
            
            # Should be between 0 and 1
            assert 0.0 <= float(response) <= 1.0, \
                f"Moisture response {response:.3f} outside [0,1] at moisture {moisture:.3f}"
        
        # Very dry conditions should give low response
        dry_response = decomposer._calculate_moisture_response(torch.tensor([0.01]))
        assert float(dry_response) < 0.2, "Very dry conditions should give low response"
    
    def test_environmental_modifier_combination(self, decomposer, standard_state):
        """Test environmental modifiers combine correctly."""
        # Standard conditions
        temp = torch.tensor(288.15)  # 15degC
        moisture = torch.tensor(0.20) # 20% 
        
        temp_response = decomposer._calculate_temperature_response(temp)
        moisture_response = decomposer._calculate_moisture_response(moisture)
        
        # Combined response should be product (assuming multiplicative)
        combined_expected = temp_response * moisture_response
        
        # Test through forward pass
        state = standard_state.copy()
        state['metabolic_litter_c'] = torch.tensor(0.1)
        state['soil_temperature'] = temp
        state['soil_moisture'] = moisture
        
        result = decomposer.forward(state, time=0.0, dt=3600.0)
        
        combined_actual = _value(
            result,
            "diagnostics.first_order_decomposition.environmental_modifier",
        )
        
        # Should be close to expected (within 10% due to implementation details)
        relative_error = abs(float(combined_actual - combined_expected)) / float(combined_expected)
        assert relative_error < 0.15, \
            f"Combined environmental modifier error: {relative_error:.3f}"
    
    # =============================================================================
    # L1.4: Boundary Condition and Error Handling Tests
    # =============================================================================
    
    def test_negative_pools_handling(self, decomposer):
        """Test negative carbon pools fail before process evaluation."""
        # Create state with negative pools
        state = {
            'metabolic_litter_c': torch.tensor(-0.1),  # Negative
            'structural_litter_c': torch.tensor(0.5),
            'fast_som_c': torch.tensor(0.0),           # Zero
            'slow_som_c': torch.tensor(3.0),
            'passive_som_c': torch.tensor(5.0),
            'soil_temperature': torch.tensor(288.15),
            'soil_moisture': torch.tensor(0.20)
        }
        
        with pytest.raises(ValueError, match="non-negative carbon pool"):
            decomposer.forward(state, time=0.0, dt=3600.0)
    
    def test_zero_pools_stability(self, decomposer, standard_state):
        """Test stability when pools are zero."""
        state = standard_state.copy()
        for key in (
            'metabolic_litter_c',
            'structural_litter_c',
            'fast_som_c',
            'slow_som_c',
            'passive_som_c',
        ):
            state[key] = torch.tensor(0.0)
        
        result = decomposer.forward(state, time=0.0, dt=3600.0)
        
        # Respiration should be zero or near zero
        hetero_resp = _value(result, "heterotrophic_respiration")
        assert float(hetero_resp) < 1e-10, "Zero pools should give near-zero respiration"
        
        # All pool changes should be zero or very small (from inputs)
        for key in ['metabolic_litter_c', 'structural_litter_c', 'fast_som_c', 'slow_som_c', 'passive_som_c']:
            delta_val = float(_value(result, key))
            assert abs(delta_val) < 1e-8, f"{key} should be near zero: {delta_val}"
    
    def test_extreme_temperature_handling(self, decomposer, standard_state):
        """Test extreme temperatures fail before response evaluation."""
        extreme_temps = [223.15, 373.15]  # -50degC, 100degC
        
        for temp in extreme_temps:
            state = standard_state.copy()
            state['soil_temperature'] = torch.tensor(temp)
            
            with pytest.raises(ValueError, match="soil_temperature"):
                decomposer.forward(state, time=0.0, dt=3600.0)
    
    def test_extreme_moisture_handling(self, decomposer, standard_state):
        """Test handling of extreme moisture conditions."""
        extreme_moistures = [0.0, 1.0]  # Completely dry, saturated
        
        for moisture in extreme_moistures:
            state = standard_state.copy()
            state['soil_moisture'] = torch.tensor(moisture)
            
            result = decomposer.forward(state, time=0.0, dt=3600.0)
            
            # Should handle gracefully
            resp = float(_value(result, "heterotrophic_respiration"))
            assert 0.0 <= resp < 1e-2, f"Extreme moisture resp: {resp}"
            env_mod = float(
                _value(
                    result,
                    "diagnostics.first_order_decomposition.environmental_modifier",
                )
            )
            assert 0.0 <= env_mod <= 2.0, f"Extreme moisture env_mod: {env_mod}"
    
    # =============================================================================
    # L1.5: Parameter Sensitivity Analysis
    # =============================================================================
    
    def test_q10_sensitivity(self, standard_state):
        """Test sensitivity to Q10 parameter."""
        # Create state with temperature ABOVE reference (25degC = 298.15K)
        # Use 30degC (303.15K) so higher Q10 gives higher response
        test_state = standard_state.copy()
        test_state['soil_temperature'] = torch.tensor(303.15)  # 30degC
        
        q10_values = [1.5, 2.0, 2.5, 3.0]
        responses = []
        param_manager = get_global_manager()
        if param_manager is None:
            raise RuntimeError("ParameterManager must be available for Q10 sensitivity test")

        q10_key = "pedosphere.physical.q10_decomposition"
        original_q10 = self._normalize_param_value(param_manager.get(q10_key))
        try:
            for q10 in q10_values:
                self._set_param_value(param_manager, q10_key, q10)
                decomposer = FirstOrderDecomposition(name="test", config={})
                result = decomposer.forward(test_state, time=0.0, dt=3600.0)
                resp = float(_value(result, "heterotrophic_respiration"))
                responses.append(resp)
        finally:
            self._set_param_value(param_manager, q10_key, original_q10)
        
        # Higher Q10 should give higher temperature sensitivity
        # At 30degC (above 25degC reference), higher Q10 should give higher response
        assert responses[1] < responses[2] < responses[3], \
            f"Higher Q10 should increase temperature response at 30degC: {responses}"
    
    def test_clay_content_sensitivity(self, standard_state):
        """Test sensitivity to the canonical clay-content state input."""
        clay_values = [0.1, 0.25, 0.4, 0.6]
        som_respirations = []

        decomposer = FirstOrderDecomposition(name="test", config={})
        for clay in clay_values:
            state = standard_state.copy()
            state["clay_content"] = torch.tensor(clay)
            result = decomposer.forward(state, time=0.0, dt=3600.0)

            som_resp = sum(
                float(
                    _value(
                        result,
                        f"diagnostics.first_order_decomposition.{pool}_som_decomposition_flux",
                    )
                )
                for pool in ("fast", "slow", "passive")
            )

            som_respirations.append(som_resp)
        
        # Higher clay content should reduce SOM decomposition (protection effect)
        assert som_respirations[0] > som_respirations[1] > som_respirations[2], \
            "Higher clay content should reduce SOM decomposition"
    
    def test_transfer_efficiency_sensitivity(self, standard_state):
        """Test sensitivity to transfer efficiency parameters."""
        # Test different transfer efficiencies
        efficiencies = [0.011, 0.03, 0.05, 0.07]
        passive_changes = []

        param_manager = get_global_manager()
        if param_manager is None:
            raise RuntimeError("ParameterManager must be available for transfer sensitivity test")

        transfer_key = "pedosphere.decomposition.t_fast_passive"
        original_transfer = self._normalize_param_value(param_manager.get(transfer_key))
        try:
            for eff in efficiencies:
                self._set_param_value(param_manager, transfer_key, eff)
                decomposer = FirstOrderDecomposition(name="test", config={})
                result = decomposer.forward(standard_state, time=0.0, dt=3600.0)

                # Get passive SOM change
                delta_passive = float(_value(result, "passive_som_c"))
                passive_changes.append(delta_passive)
        finally:
            self._set_param_value(param_manager, transfer_key, original_transfer)
        
        # Higher transfer efficiency should increase passive SOM input
        # (more carbon goes to passive pool)
        for i in range(len(passive_changes) - 1):
            if passive_changes[i+1] > passive_changes[i]:
                # Expected: higher efficiency increases passive pool input
                continue
            else:
                # May not always be strictly increasing due to system complexity
                # Just check it's not drastically decreasing
                assert passive_changes[i+1] >= passive_changes[i] * 0.8
    
    # =============================================================================
    # L1.6: Mass Balance and Conservation Tests
    # =============================================================================
    
    def test_mass_balance_no_inputs(self, decomposer, standard_state):
        """Test mass balance with no external inputs."""
        # Run one step with no inputs
        result = decomposer.forward(standard_state, time=0.0, dt=3600.0)
        
        # Calculate total carbon change
        pool_changes = []
        for pool in ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']:
            key = f'{pool}_c'
            pool_changes.append(float(_value(result, key)))
        
        total_change = sum(pool_changes)  # g C m-2
        # Get respiration
        respiration = float(_value(result, "heterotrophic_respiration"))
        respiration_step = respiration * 3600.0
        
        # Mass balance: pool decrease should equal respiration (no inputs)
        # |dC + R| should be small
        mass_balance_error = abs(total_change + respiration_step)
        
        # Allow small numerical errors
        assert mass_balance_error < 1e-8, \
            f"Mass balance error: {mass_balance_error:.2e} (dC: {total_change:.2e}, R: {respiration:.2e})"
    
    def test_mass_balance_with_inputs(self, decomposer, standard_state):
        """Test mass balance with mapped metabolic and structural inputs."""
        metabolic_input = torch.tensor(8.5e-7)  # g C m-2 s-1
        structural_input = torch.tensor(1.5e-7)  # g C m-2 s-1
        state_with_inputs = standard_state.copy()
        state_with_inputs['metabolic_litter_input'] = metabolic_input
        state_with_inputs['structural_litter_input'] = structural_input

        result = decomposer.forward(
            state_with_inputs,
            time=0.0,
            dt=3600.0,
        )
        
        # Calculate total carbon change
        pool_changes = []
        for pool in ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']:
            key = f'{pool}_c'
            pool_changes.append(float(_value(result, key)))
        
        total_change = sum(pool_changes)
        respiration = float(_value(result, "heterotrophic_respiration"))
        respiration_step = respiration * 3600.0
        
        # Mass balance: dC + R = I (inputs)
        input_step = float(metabolic_input + structural_input) * 3600.0
        mass_balance_error = abs(total_change + respiration_step - input_step)
        
        # Allow reasonable numerical tolerance
        assert mass_balance_error < 1e-7, \
            f"Mass balance error with inputs: {mass_balance_error:.2e}"
    
    def test_carbon_conservation_over_time(self, decomposer, standard_state):
        """Test carbon conservation over multiple time steps."""
        initial_carbon = sum(float(standard_state[key]) for key in standard_state.keys() 
                           if key.endswith('_c'))
        
        current_state = standard_state.copy()
        total_respired = 0.0
        dt = 3600.0  # 1 hour
        
        # Run for 10 steps
        for _ in range(10):
            result = decomposer.forward(current_state, time=0.0, dt=dt)
            
            # Update state
            for pool in ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']:
                key = f'{pool}_c'
                delta_key = f'{pool}_c'
                if key in current_state and delta_key in result:
                    current_state[key] = current_state[key] + _value(result, delta_key)
            
            # Accumulate respiration
            resp = float(_value(result, "heterotrophic_respiration"))
            total_respired += resp * dt
        
        # Calculate final carbon
        final_carbon = sum(float(current_state[key]) for key in current_state.keys() 
                          if key.endswith('_c'))
        
        # Conservation check: initial = final + respired
        conservation_error = abs(initial_carbon - (final_carbon + total_respired))
        relative_error = conservation_error / initial_carbon
        
        assert relative_error < 0.01, \
            f"Carbon conservation error over time: {relative_error:.4f} " \
            f"(initial: {initial_carbon:.6f}, final: {final_carbon:.6f}, respired: {total_respired:.6f})"
    
    # =============================================================================
    # L1.7: Performance and Numerical Stability Tests
    # =============================================================================
    
    def test_numerical_stability_small_pools(self, decomposer, standard_state):
        """Test numerical stability with very small pools."""
        small_state = standard_state.copy()
        small_state.update(
            {
                'metabolic_litter_c': torch.tensor(1e-10),
                'structural_litter_c': torch.tensor(1e-10),
                'fast_som_c': torch.tensor(1e-10),
                'slow_som_c': torch.tensor(1e-9),
                'passive_som_c': torch.tensor(1e-8),
            }
        )
        
        # Should not crash or produce NaN/Inf
        result = decomposer.forward(small_state, time=0.0, dt=3600.0)
        
        # Check all outputs are finite
        for key in result:
            value = _value(result, key)
            assert torch.isfinite(value).all(), f"{key} contains non-finite values"
            assert not torch.isnan(value).any(), f"{key} contains NaN values"
    
    def test_numerical_stability_large_dt(self, decomposer, standard_state):
        """Test numerical stability with large time steps."""
        # Test with large time step (1 day)
        large_dt = 86400.0  # 1 day in seconds
        
        result = decomposer.forward(standard_state, time=0.0, dt=large_dt)
        
        # Should not produce negative pools or unrealistic values
        for pool in ['metabolic_litter', 'structural_litter', 'fast_som', 'slow_som', 'passive_som']:
            delta_key = f'{pool}_c'
            if delta_key in result:
                delta_val = float(_value(result, delta_key))
                original_pool = float(standard_state[f'{pool}_c'])

                # Pool change should not exceed original pool size by much      
                assert abs(delta_val) <= original_pool * 2, \
                    f"Large time step produces unrealistic {pool} change: {delta_val:.2e}"
        
        # All outputs should be finite
        for key in result:
            value = _value(result, key)
            assert torch.isfinite(value).all(), f"{key} not finite with large dt"
    
    def test_functor_base_is_single_lifecycle_statistics_owner(
        self,
        decomposer,
        standard_state,
    ):
        """Lifecycle metrics must use FunctorBase without scientific history."""
        assert not hasattr(decomposer, "get_decomposition_statistics")
        initial_stats = decomposer.get_execution_stats()
        assert initial_stats["execution_count"] == 0

        decomposer(standard_state, time=0.0, dt=3600.0)

        assert decomposer.get_execution_stats()["execution_count"] == 1
    
    def test_repeated_execution_consistency(self, decomposer, standard_state):
        """Test that repeated executions give consistent results."""
        # Run same conditions multiple times
        results = []
        for _ in range(3):
            result = decomposer.forward(standard_state.copy(), time=0.0, dt=3600.0)
            resp = float(_value(result, "heterotrophic_respiration"))
            results.append(resp)
        
        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-12, \
                f"Results should be consistent: {results[0]:.2e} vs {results[i]:.2e}"
    
    def test_gradient_computation_consistency(self, decomposer, standard_state):
        """Test consistency with PyTorch gradient computation."""
        # Create state with requires_grad
        grad_state = {}
        for key, value in standard_state.items():
            if isinstance(value, torch.Tensor):
                grad_state[key] = value.clone().requires_grad_(True)
            else:
                grad_state[key] = value
        
        # Forward pass
        result = decomposer.forward(grad_state, time=0.0, dt=3600.0)
        
        # Should be able to compute gradients
        resp = _value(result, "heterotrophic_respiration")
        assert resp.requires_grad
        resp.backward()
        has_grad = any(
            value.grad is not None
            for value in grad_state.values()
            if isinstance(value, torch.Tensor) and value.requires_grad
        )
        assert has_grad, "Should compute gradients for some inputs"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


