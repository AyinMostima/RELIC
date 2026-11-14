"""
L2 Integration Tests for Nitrogen Cycling Processes

Tests the integration and coupling between different nitrogen cycling processes
in the pedosphere module, including mineralization, nitrification, denitrification,
stoichiometry constraints, and nitrogen leaching.
"""

import builtins
import warnings
from typing import Any, Dict

import pytest
import torch

# Import support classes
from relic.core.abstractions import StateDict
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem
from relic.scientific.biosphere.phenology.models import StaticPhenology
from relic.scientific.pedosphere.decomposition import FirstOrderDecomposition
from relic.scientific.pedosphere.interface_mapping import LitterfallInterfaceMapper
from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess

# Import nitrogen cycling processes
from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrogen import NitrogenCycling
from relic.scientific.pedosphere.nutrient_cycling.nitrogen_leaching import NitrogenLeachingProcess
from relic.scientific.pedosphere.nutrient_cycling.root_nutrient_uptake import RootNutrientUptake
from relic.scientific.pedosphere.nutrient_cycling.stoichiometry import StoichiometryConstraint


def _to_float(value: torch.Tensor) -> float:
    if isinstance(value, UpdateItem):
        value = value.value
    return builtins.float(torch.as_tensor(value).detach().item())


def _update_value(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


class TestNitrogenCyclingL2Integration:
    """L2 Integration Tests for complete nitrogen cycling processes."""
    
    @pytest.fixture
    def default_configs(self) -> Dict[str, Dict]:
        """Default configurations for all nitrogen processes."""
        return {
            'mineralization': {
                'q10': 2.2,
                'reference_temperature': 298.15,
                'minimum_temperature': 273.15,
                'maximum_temperature': 323.15,
                'optimal_moisture': 0.6,
                'minimum_moisture': 0.1
            },
            'nitrification': {
                'vmax': 1e-6,
                'km_nh4': 5e-4,
                'q10': 2.0,
                'reference_temperature': 298.15,
                'ph_optimum': 8.0,
                'ph_range': 2.0,
                'n2o_factor': 0.02
            },
            'denitrification': {
                'vmax': 8e-7,
                'km_no3': 1e-3,
                'q10': 1.8,
                'reference_temperature': 298.15,
                'anaerobic_threshold': 0.8,
                'ph_optimum': 7.0,
                'ph_range': 2.5
            },
            'leaching': {
                'adsorption_coefficient': 0.1,
                'max_flux_fraction': 0.5,
                'leaching_time_constant': 3600.0
            },
            'stoichiometry': {
                'target_cn_ratio': 15.0,
                'min_cn_ratio': 5.0,
                'max_cn_ratio': 50.0,
                'correction_rate': 0.1
            }
        }
    
    @pytest.fixture
    def nitrogen_processes(self, default_configs) -> Dict[str, Any]:
        """Create all nitrogen cycling processes."""
        return {
            'mineralization': MineralizationProcess("mineralization", default_configs['mineralization']),
            'nitrification': NitrificationProcess("nitrification", default_configs['nitrification']),
            'denitrification': DenitrificationProcess("denitrification", default_configs['denitrification']),
            'leaching': NitrogenLeachingProcess("leaching", default_configs['leaching']),
            'stoichiometry': StoichiometryConstraint("stoichiometry", default_configs['stoichiometry'])
        }
    
    @pytest.fixture
    def typical_soil_state(self) -> StateDict:
        """Typical agricultural soil state."""
        return {
            'litter_c': torch.tensor(100.0),
            'fast_som_c': torch.tensor(500.0),
            'slow_som_c': torch.tensor(2000.0),
            'passive_som_c': torch.tensor(5000.0),
            'litter_n': torch.tensor(5.0),
            'fast_som_n': torch.tensor(25.0),
            'slow_som_n': torch.tensor(100.0),
            'passive_som_n': torch.tensor(250.0),
            'nh4_nitrogen': torch.tensor(0.05),
            'no3_nitrogen': torch.tensor(0.02),
            'organic_nitrogen': torch.tensor(0.03),
            'organic_nitrogen_input': torch.tensor(0.0),
            'microbial_nitrogen': torch.tensor(0.01),
            'soil_temperature': torch.tensor(298.15),
            'soil_moisture': torch.tensor(0.3),
            'water_filled_pore_space': torch.tensor(0.5),
            'soil_ph': torch.tensor(6.5),
            'carbon_decomposition_rate': torch.tensor(1e-6),
            'oxygen_fraction': torch.tensor(0.15),
            'clay_fraction': torch.tensor(0.2),
            'drainage_flux': torch.tensor(1e-6),
            # Required forcing inputs (no environment or geometry default is substituted).
            'labile_carbon': torch.tensor(0.1),  # g C m-2
            'soil_depth': torch.tensor(1.0),  # m
            'bulk_density': torch.tensor(1300.0),  # kg m-3
        }
    
    def test_nitrogen_cascade_sequence(self, nitrogen_processes, typical_soil_state):
        """Test the complete nitrogen cascade: mineralization 鈫?nitrification 鈫?denitrification."""
        
        # Initial state
        initial_state = typical_soil_state.copy()
        dt = 3600.0  # 1 hour
        
        # Step 1: Mineralization (organic N 鈫?NH4+)
        mineralization = nitrogen_processes['mineralization']
        min_result = mineralization.forward(initial_state, 0.0, dt)

        # Update state with mineralization results
        state_after_min = initial_state.copy()
        state_after_min['nh4_nitrogen'] += _update_value(
            min_result.get('nh4_nitrogen', torch.tensor(0.0))
        )
        state_after_min['organic_nitrogen'] += _update_value(
            min_result['organic_nitrogen']
        )
        state_after_min['microbial_nitrogen'] += _update_value(
            min_result['microbial_nitrogen']
        )

        # Verify mineralization produced NH4+
        nh4_increase = _to_float((state_after_min['nh4_nitrogen'] - initial_state['nh4_nitrogen']))
        assert nh4_increase >= -1e-10, "Mineralization should not decrease NH4+ significantly"

        # Also check that the process actually ran (gross mineralization > 0)
        gross_min = _to_float(min_result.get('gross_mineralization_flux', torch.tensor(0.0)))
        assert gross_min > 0, f"Gross mineralization should be positive: {gross_min}"
        
        # Step 2: Nitrification (NH4+ 鈫?NO3-)
        nitrification = nitrogen_processes['nitrification']
        nit_result = nitrification.forward(state_after_min, dt, dt)
        
        # Update state with nitrification results
        state_after_nit = state_after_min.copy()
        state_after_nit['nh4_nitrogen'] += _update_value(
            nit_result.get('nh4_nitrogen', torch.tensor(0.0))
        )
        state_after_nit['no3_nitrogen'] += _update_value(
            nit_result.get('no3_nitrogen', torch.tensor(0.0))
        )
        
        # Verify nitrification converted NH4+ 鈫?NO3-
        nh4_decrease = _to_float((state_after_min['nh4_nitrogen'] - state_after_nit['nh4_nitrogen']))
        no3_increase = _to_float((state_after_nit['no3_nitrogen'] - state_after_min['no3_nitrogen']))
        
        # Check the process rates directly from the result
        nit_rate = _to_float(nit_result.get('nitrification_flux', torch.tensor(0.0)))
        delta_nh4 = _to_float(nit_result.get('nh4_nitrogen', torch.tensor(0.0)))
        delta_no3 = _to_float(nit_result.get('no3_nitrogen', torch.tensor(0.0)))
        
        assert nit_rate > 0, f"Nitrification rate should be positive: {nit_rate}"
        assert delta_nh4 < 0, f"NH4 delta should be negative: {delta_nh4}"
        assert delta_no3 > 0, f"NO3 delta should be positive: {delta_no3}"
        
        # Step 3: Denitrification (NO3- 鈫?N2O + N2)
        # Make conditions anaerobic
        state_after_nit['oxygen_fraction'] = torch.tensor(0.05)  # Low oxygen
        
        denitrification = nitrogen_processes['denitrification']
        denit_result = denitrification.forward(state_after_nit, 2*dt, dt)
        
        # Verify denitrification consumed NO3-
        no3_loss = _to_float(denit_result.get('no3_nitrogen', torch.tensor(0.0)))
        assert no3_loss <= 0, "Denitrification should consume NO3-"
        
    def test_nitrogen_leaching_integration(self, nitrogen_processes, typical_soil_state):
        """Test nitrogen leaching integration with other N processes."""
        
        state = typical_soil_state.copy()
        state['no3_nitrogen'] = torch.tensor(0.1)  # Higher NO3- for leaching
        state['drainage_flux'] = torch.tensor(5e-6)  # Higher drainage
        
        dt = 3600.0
        
        # Run leaching process
        leaching = nitrogen_processes['leaching']
        leach_result = leaching.forward(state, 0.0, dt)
        
        # Verify leaching removes NO3-
        no3_loss = _to_float(leach_result['no3_nitrogen'])
        leach_flux = _to_float(leach_result['nitrogen_leaching_flux'])
        
        assert no3_loss < 0, "Leaching should remove NO3- from soil"
        assert leach_flux > 0, "Leaching flux should be positive"
        assert abs(no3_loss + leach_flux * dt) < 1e-12, "Mass balance: loss should equal flux integrated over dt"
        
    def test_stoichiometry_constraint_integration(self, nitrogen_processes, typical_soil_state):
        """Test stoichiometry constraint integration with other processes."""
        
        # Create C:N imbalanced state
        state = typical_soil_state.copy()
        state['fast_som_c'] = torch.tensor(1000.0)  # High C
        state['fast_som_n'] = torch.tensor(10.0)    # Low N (C:N = 100, too high)
        state['diagnostics.carbon_fluxes'] = {
            'delta_fast_som_c': torch.tensor(1.0e-6),
        }
        state['diagnostics.nitrogen_fluxes'] = {
            'delta_fast_som_n': torch.tensor(1.0e-8),
        }
        
        dt = 3600.0
        
        # Apply stoichiometry constraint
        stoichiometry = nitrogen_processes['stoichiometry']
        stoich_result = stoichiometry.forward(state, 0.0, dt)
        
        # Verify constraint tried to balance C:N ratio
        c_change = stoich_result.get(
            'diagnostics.stoichiometry_corrected_delta_fast_som_c',
            torch.tensor(0.0),
        )
        n_change = stoich_result.get(
            'diagnostics.stoichiometry_corrected_delta_fast_som_n',
            torch.tensor(0.0),
        )
        
        # Should either reduce C or increase N (or both) to lower C:N ratio
        assert _to_float(c_change) <= 0 or _to_float(n_change) >= 0, \
            "Stoichiometry should reduce C:N ratio when too high"
            
    def test_multi_process_mass_conservation(self, nitrogen_processes, typical_soil_state):
        """Test mass conservation when running multiple N processes simultaneously."""
        
        state = typical_soil_state.copy()
        state['diagnostics.carbon_fluxes'] = {
            'delta_fast_som_c': torch.tensor(1.0e-6),
        }
        state['diagnostics.nitrogen_fluxes'] = {
            'delta_fast_som_n': torch.tensor(1.0e-8),
        }
        dt = 3600.0
        
        # Track initial total nitrogen
        initial_n_pools = [
            'litter_n', 'fast_som_n', 'slow_som_n', 'passive_som_n',
            'nh4_nitrogen', 'no3_nitrogen', 'organic_n'
        ]
        
        initial_total_n = sum(_to_float(state.get(pool, torch.tensor(0.0))) for pool in initial_n_pools)
        
        # Run all processes
        all_results = {}
        for name, process in nitrogen_processes.items():
            all_results[name] = process.forward(state, 0.0, dt)
        
        # Calculate total N fluxes (excluding leaching which removes N from system)
        total_n_flux_in = 0.0
        total_n_flux_out = 0.0
        
        for name, result in all_results.items():
            if name == 'leaching':
                # Leaching removes N from the system
                total_n_flux_out += _to_float(result.get('nitrogen_leaching_flux', torch.tensor(0.0)))
            else:
                # Other processes transform N within the system
                for key, value in result.items():
                    if 'delta' in key and 'nitrogen' in key or 'delta' in key and any(pool in key for pool in ['nh4', 'no3']):
                        flux_val = _to_float(value)
                        if flux_val > 0:
                            total_n_flux_in += flux_val
                        else:
                            total_n_flux_out += abs(flux_val)
        
        # Internal transformations should balance (excluding system losses)
        internal_balance = total_n_flux_in - total_n_flux_out
        
        # Allow for system losses (leaching) but internal processes should conserve mass
        leach_loss = _to_float(all_results.get('leaching', {}).get('nitrogen_leaching_flux', torch.tensor(0.0)))
        expected_balance = -leach_loss  # Negative because it's a loss
        
        # Mass conservation check (allow small numerical errors typical in ecological models)
        balance_error = abs(internal_balance - expected_balance)
        assert balance_error < 1e-3, f"Mass conservation error: {balance_error} (acceptable for ecological models)"
        
    def test_environmental_response_integration(self, nitrogen_processes, typical_soil_state):
        """Test that all N processes respond appropriately to environmental changes."""
        
        base_state = typical_soil_state.copy()
        dt = 3600.0
        
        # Test temperature response
        cold_state = base_state.copy()
        cold_state['soil_temperature'] = torch.tensor(278.15)  # 5掳C
        
        hot_state = base_state.copy()
        hot_state['soil_temperature'] = torch.tensor(313.15)  # 40掳C
        
        # Run processes at different temperatures
        temp_responses = {}
        for temp_name, state in [('cold', cold_state), ('base', base_state), ('hot', hot_state)]:
            temp_responses[temp_name] = {}
            for process_name in ['mineralization', 'nitrification', 'denitrification']:
                process = nitrogen_processes[process_name]
                result = process.forward(state, 0.0, dt)
                # Extract the canonical main flux from each process.
                if process_name == 'mineralization':
                    temp_responses[temp_name][process_name] = _to_float(
                        result['gross_mineralization_flux']
                    )
                elif process_name == 'nitrification':
                    temp_responses[temp_name][process_name] = _to_float(
                        result['nitrification_flux']
                    )
                elif process_name == 'denitrification':
                    temp_responses[temp_name][process_name] = _to_float(
                        result['denitrification_flux']
                    )
        
        # Verify temperature responses (rates should generally increase with temperature up to optimum)
        for process_name in ['mineralization', 'nitrification', 'denitrification']:
            base_rate = temp_responses['base'][process_name]
            hot_rate = temp_responses['hot'][process_name]
            
            # At moderate temperatures, rate should generally increase with temperature
            if base_rate > 0:
                assert hot_rate >= base_rate, f"{process_name} should show positive temperature response"
                
    def test_ph_response_integration(self, nitrogen_processes, typical_soil_state):
        """Test pH response integration across N processes."""
        
        base_state = typical_soil_state.copy()
        dt = 3600.0
        
        # Test different pH conditions
        acidic_state = base_state.copy()
        acidic_state['soil_ph'] = torch.tensor(4.5)  # Acidic
        
        alkaline_state = base_state.copy() 
        alkaline_state['soil_ph'] = torch.tensor(8.5)  # Alkaline
        
        ph_responses = {}
        for ph_name, state in [('acidic', acidic_state), ('base', base_state), ('alkaline', alkaline_state)]:
            ph_responses[ph_name] = {}
            
            # Test nitrification (prefers slightly alkaline)
            nitrification = nitrogen_processes['nitrification']
            nit_result = nitrification.forward(state, 0.0, dt)
            ph_responses[ph_name]['nitrification'] = _to_float(nit_result.get('nitrification_flux', torch.tensor(0.0)))
            
            # Test denitrification (prefers near-neutral to slightly alkaline)
            denitrification = nitrogen_processes['denitrification']
            # Make conditions anaerobic for denitrification
            denit_state = state.copy()
            denit_state['oxygen_fraction'] = torch.tensor(0.05)
            
            denit_result = denitrification.forward(denit_state, 0.0, dt)
            ph_responses[ph_name]['denitrification'] = _to_float(denit_result.get('denitrification_flux', torch.tensor(0.0)))
        
        # Verify pH responses
        # Nitrification should be higher at alkaline pH than acidic pH
        if ph_responses['base']['nitrification'] > 0:
            assert ph_responses['alkaline']['nitrification'] >= ph_responses['acidic']['nitrification'], \
                "Nitrification should prefer alkaline conditions over acidic"
                
    def test_moisture_response_integration(self, nitrogen_processes, typical_soil_state):
        """Test soil moisture response integration."""
        
        base_state = typical_soil_state.copy()
        dt = 3600.0
        
        # Test different moisture conditions
        dry_state = base_state.copy()
        dry_state['soil_moisture'] = torch.tensor(0.1)  # Dry
        
        wet_state = base_state.copy()
        wet_state['soil_moisture'] = torch.tensor(0.6)  # Wet
        
        moisture_responses = {}
        for moisture_name, state in [('dry', dry_state), ('base', base_state), ('wet', wet_state)]:
            moisture_responses[moisture_name] = {}
            
            # Test mineralization (should increase with moisture up to optimum)
            mineralization = nitrogen_processes['mineralization']
            min_result = mineralization.forward(state, 0.0, dt)
            moisture_responses[moisture_name]['mineralization'] = _to_float(
                min_result['gross_mineralization_flux']
            )
            
            # Test leaching (should increase with moisture/drainage)
            # Adjust drainage based on moisture
            leach_state = state.copy()
            leach_state['drainage_flux'] = state['soil_moisture'] * 1e-6  # Higher moisture -> higher drainage
            
            leaching = nitrogen_processes['leaching']
            leach_result = leaching.forward(leach_state, 0.0, dt)
            moisture_responses[moisture_name]['leaching'] = _to_float(leach_result.get('nitrogen_leaching_flux', torch.tensor(0.0)))
        
        # Verify moisture responses
        # Mineralization should increase with moisture (up to optimum)
        assert moisture_responses['wet']['mineralization'] >= moisture_responses['dry']['mineralization'], \
            "Mineralization should increase with soil moisture"
            
        # Leaching should increase with moisture
        leaching_epsilon = 1e-15
        assert moisture_responses['wet']['leaching'] + leaching_epsilon >= moisture_responses['dry']['leaching'], \
            "Leaching should increase with soil moisture"
            
    def test_oxygen_limitation_integration(self, nitrogen_processes, typical_soil_state):
        """Test oxygen limitation effects on N processes."""
        
        base_state = typical_soil_state.copy()
        dt = 3600.0
        
        # Aerobic conditions
        aerobic_state = base_state.copy()
        aerobic_state['oxygen_fraction'] = torch.tensor(0.18)  # 18% O2 (aerobic)
        
        # Anaerobic conditions
        anaerobic_state = base_state.copy()
        anaerobic_state['oxygen_fraction'] = torch.tensor(0.02)  # 2% O2 (anaerobic)
        
        # Test nitrification (requires oxygen)
        nitrification = nitrogen_processes['nitrification']
        
        aerobic_nit = nitrification.forward(aerobic_state, 0.0, dt)
        anaerobic_nit = nitrification.forward(anaerobic_state, 0.0, dt)
        
        aerobic_nit_rate = _to_float(aerobic_nit.get('nitrification_flux', torch.tensor(0.0)))
        anaerobic_nit_rate = _to_float(anaerobic_nit.get('nitrification_flux', torch.tensor(0.0)))
        
        # Nitrification should be higher under aerobic conditions
        assert aerobic_nit_rate >= anaerobic_nit_rate, \
            "Nitrification should be higher under aerobic conditions"
            
        # Test denitrification (requires anaerobic conditions)
        denitrification = nitrogen_processes['denitrification']
        
        aerobic_denit = denitrification.forward(aerobic_state, 0.0, dt)
        anaerobic_denit = denitrification.forward(anaerobic_state, 0.0, dt)
        
        aerobic_denit_rate = _to_float(aerobic_denit.get('denitrification_flux', torch.tensor(0.0)))
        anaerobic_denit_rate = _to_float(anaerobic_denit.get('denitrification_flux', torch.tensor(0.0)))
        
        # Denitrification should be higher under anaerobic conditions
        assert anaerobic_denit_rate >= aerobic_denit_rate, \
            "Denitrification should be higher under anaerobic conditions"
            
    def test_process_coupling_feedback_loops(self, nitrogen_processes, typical_soil_state):
        """Test feedback loops between coupled N processes."""
        
        state = typical_soil_state.copy()
        dt = 3600.0
        
        # Run processes in sequence multiple times to see coupling effects
        initial_nh4 = _to_float(state['nh4_nitrogen'])
        initial_no3 = _to_float(state['no3_nitrogen'])
        
        for step in range(3):
            # Mineralization increases NH4+
            mineralization = nitrogen_processes['mineralization']
            min_result = mineralization.forward(state, step * dt, dt)
            for pool in (
                'organic_nitrogen',
                'nh4_nitrogen',
                'microbial_nitrogen',
            ):
                state[pool] += _update_value(min_result[pool])
            
            # Nitrification converts NH4+ 鈫?NO3-
            nitrification = nitrogen_processes['nitrification']
            nit_result = nitrification.forward(state, step * dt, dt)
            state['nh4_nitrogen'] += _update_value(
                nit_result.get('nh4_nitrogen', torch.tensor(0.0))
            )
            state['no3_nitrogen'] += _update_value(
                nit_result.get('no3_nitrogen', torch.tensor(0.0))
            )
            
            # Leaching removes NO3-
            leaching = nitrogen_processes['leaching']
            leach_result = leaching.forward(state, step * dt, dt)
            state['no3_nitrogen'] += _update_value(
                leach_result.get('no3_nitrogen', torch.tensor(0.0))
            )
        
        # Verify coupled effects
        final_nh4 = _to_float(state['nh4_nitrogen'])
        final_no3 = _to_float(state['no3_nitrogen'])
        
        # After multiple steps, we should see the effects of the nitrogen cascade
        nh4_net_change = final_nh4 - initial_nh4
        no3_net_change = final_no3 - initial_no3
        
        # Net changes should reflect the balance of production and consumption
        assert nh4_net_change != 0 or no3_net_change != 0, \
            "Coupled processes should produce net changes in N pools"
            
    def test_process_execution_order_independence(self, nitrogen_processes, typical_soil_state):
        """Test that process execution order doesn't affect mass conservation."""
        
        # Test two different execution orders
        state1 = typical_soil_state.copy()
        state2 = typical_soil_state.copy()
        dt = 3600.0
        
        # Order 1: mineralization 鈫?nitrification 鈫?denitrification
        processes_order1 = ['mineralization', 'nitrification', 'denitrification']
        
        # Order 2: denitrification 鈫?nitrification 鈫?mineralization  
        processes_order2 = ['denitrification', 'nitrification', 'mineralization']
        
        results1 = {}
        results2 = {}
        
        # Execute in order 1
        for process_name in processes_order1:
            process = nitrogen_processes[process_name]
            if process_name == 'denitrification':
                state1['oxygen_fraction'] = torch.tensor(0.05)  # Make anaerobic
            results1[process_name] = process.forward(state1, 0.0, dt)
        
        # Execute in order 2
        for process_name in processes_order2:
            process = nitrogen_processes[process_name]
            if process_name == 'denitrification':
                state2['oxygen_fraction'] = torch.tensor(0.05)  # Make anaerobic
            results2[process_name] = process.forward(state2, 0.0, dt)
        
        # Both orders should produce valid results (though potentially different magnitudes)
        # The key is that mass should be conserved in both cases
        for order, results in [('order1', results1), ('order2', results2)]:
            total_n_delta = 0.0
            for process_name, result in results.items():
                for key, value in result.items():
                    if 'delta' in key and ('nitrogen' in key or 'nh4' in key or 'no3' in key):
                        total_n_delta += _to_float(value)
            
            # Total internal nitrogen changes should be small (conservation)
            # Allow for leaching losses if present
            assert abs(total_n_delta) < 1e-3, f"Mass conservation violated in {order}: {total_n_delta}"


def test_real_simulator_nitrogen_ledger_has_exact_writers_and_no_false_warnings(caplog):
    """The canonical scheduler must compose each nitrogen increment exactly once."""
    simulator = RELICSimulator(
        {
            "max_time": 10.0,
            "time_step": 10.0,
            "conservation_checks": False,
            "strict_scale_detection": False,
        }
    )
    state = {
        "organic_nitrogen": torch.tensor(5.0),
        "organic_nitrogen_input": torch.tensor(0.0),
        "nh4_nitrogen": torch.tensor(0.1),
        "no3_nitrogen": torch.tensor(0.2),
        "microbial_nitrogen": torch.tensor(0.05),
        "available_phosphorus": torch.tensor(1.0),
        "plant_n_storage": torch.tensor(0.1),
        "plant_p_storage": torch.tensor(0.05),
        "soil_temperature": torch.tensor(288.15),
        "soil_moisture": torch.tensor(0.3),
        "water_filled_pore_space": torch.tensor(0.5),
        "soil_ph": torch.tensor(6.5),
        "carbon_decomposition_rate": torch.tensor(1.0e-6),
        "labile_carbon": torch.tensor(0.1),
        "drainage_flux": torch.tensor(1.0e-6),
        "fertilizer_n_flux": torch.tensor(1.0e-6),
        # One-layer column: depth and bulk density share the scalar pool shape.
        "soil_depth": torch.tensor(0.1),
        "bulk_density": torch.tensor(1300.0),  # kg m-3 leaching forcing
        "root_density": torch.tensor([1.0]),
        "soil_water_content": torch.tensor([0.3]),
        "plant_n_demand": torch.tensor(1.0e-6),
        "plant_p_demand": torch.tensor(1.0e-7),
        "field_capacity": torch.tensor([0.45]),
        "wilting_point": torch.tensor([0.1]),
    }
    simulator.initialize_state(state)
    for functor in (
        MineralizationProcess(),
        NitrificationProcess(),
        DenitrificationProcess(),
        NitrogenLeachingProcess(),
        RootNutrientUptake(),
        NitrogenCycling(),
    ):
        simulator.register_functor(functor, execution_group="fast")

    before = {
        key: simulator.state_manager.get_state(key).clone()
        for key in ("nh4_nitrogen", "no3_nitrogen")
    }
    with caplog.at_level("WARNING"):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            updates = simulator.step(dt=10.0)

    conservation_warnings = [
        str(item.message)
        for item in captured
        if "Conservation law" in str(item.message)
    ]
    assert conservation_warnings == []
    direction_warnings = [
        record.getMessage()
        for record in caplog.records
        if "direction" in record.getMessage()
    ]
    assert direction_warnings == []
    for key in ("nh4_nitrogen", "no3_nitrogen"):
        state_delta = simulator.state_manager.get_state(key) - before[key]
        update_value = updates[key]
        if isinstance(update_value, UpdateItem):
            update_value = update_value.value
        assert torch.allclose(state_delta, update_value)

    expected_sources = {
        "nh4_nitrogen": {
            "NitrogenCycling",
            "mineralization",
            "nitrification",
            "root_nutrient_uptake",
        },
        "no3_nitrogen": {
            "denitrification",
            "nitrification",
            "nitrogen_leaching",
            "root_nutrient_uptake",
        },
    }
    for key, expected in expected_sources.items():
        observed = {
            item["functor"]
            for item in simulator.get_variable_provenance(key)
            if item.get("role") == "increment_source"
        }
        assert observed == expected


def test_phenology_litter_decomposition_nitrogen_chain_restarts_exactly(caplog):
    """Run the canonical litter C/N boundary through all nitrogen owners.

    References:
    - Thornton et al. (2002), Agricultural and Forest Meteorology 113.
    - Parton et al. (1987), Soil Science Society of America Journal 51.
    - Manzoni and Porporato (2009), Soil Biology and Biochemistry 41.

    Mathematical expressions:
    - F_soil = 1000 * F_vegetation.
    - Delta C_veg,g + Delta C_soil + R_h * dt = 0.
    - x_N(t + dt) - x_N(t) = sum_j Delta x_N,j.

    Parameter meanings:
    - dt is the integration interval [s].
    - F is a carbon or nitrogen boundary flux [kg or g element m-2 s-1].
    - x_N is an ammonium or nitrate pool [g N m-2].
    """
    dt = 10.0
    initial_state = {
        "metabolic_litter_c": torch.tensor(100.0, dtype=torch.float64),
        "structural_litter_c": torch.tensor(200.0, dtype=torch.float64),
        "fast_som_c": torch.tensor(500.0, dtype=torch.float64),
        "slow_som_c": torch.tensor(1000.0, dtype=torch.float64),
        "passive_som_c": torch.tensor(2000.0, dtype=torch.float64),
        "soil_temperature": torch.tensor(288.15, dtype=torch.float64),
        "soil_moisture": torch.tensor(0.3, dtype=torch.float64),
        "water_filled_pore_space": torch.tensor(0.5, dtype=torch.float64),
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
        "no3_nitrogen": torch.tensor(0.2, dtype=torch.float64),
        "microbial_nitrogen": torch.tensor(0.05, dtype=torch.float64),
        "available_phosphorus": torch.tensor(1.0, dtype=torch.float64),
        "plant_n_storage": torch.tensor(0.1, dtype=torch.float64),
        "plant_p_storage": torch.tensor(0.05, dtype=torch.float64),
        "soil_ph": torch.tensor(6.5, dtype=torch.float64),
        "labile_carbon": torch.tensor(0.1, dtype=torch.float64),
        "drainage_flux": torch.tensor(1.0e-6, dtype=torch.float64),
        "fertilizer_n_flux": torch.tensor(1.0e-6, dtype=torch.float64),
        # One-layer column: depth and bulk density share the scalar pool shape.
        "soil_depth": torch.tensor(0.1, dtype=torch.float64),
        "bulk_density": torch.tensor(1300.0, dtype=torch.float64),  # kg m-3 leaching forcing
        "root_density": torch.tensor([1.0], dtype=torch.float64),
        "soil_water_content": torch.tensor([0.3], dtype=torch.float64),
        "plant_n_demand": torch.tensor(1.0e-6, dtype=torch.float64),
        "plant_p_demand": torch.tensor(1.0e-7, dtype=torch.float64),
        "field_capacity": torch.tensor([0.45], dtype=torch.float64),
        "wilting_point": torch.tensor([0.1], dtype=torch.float64),
    }

    def build_simulator() -> RELICSimulator:
        simulator = RELICSimulator(
            {
                "max_time": 2.0 * dt,
                "time_step": dt,
                "device": "cpu",
                "conservation_checks": False,
                "strict_dependencies": True,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize_state(
            {key: value.clone() for key, value in initial_state.items()}
        )
        for functor in (
            StaticPhenology(name="static_phenology"),
            LitterfallInterfaceMapper(
                name="litterfall_interface_mapper",
                config={"dependencies": ["static_phenology"]},
            ),
            FirstOrderDecomposition(
                name="first_order_decomposition",
                config={"dependencies": ["litterfall_interface_mapper"]},
            ),
            MineralizationProcess(
                name="mineralization",
                config={
                    "dependencies": [
                        "first_order_decomposition",
                        "litterfall_interface_mapper",
                    ]
                },
            ),
            NitrificationProcess(),
            DenitrificationProcess(),
            NitrogenLeachingProcess(),
            RootNutrientUptake(),
            NitrogenCycling(),
        ):
            simulator.register_functor(functor, execution_group="fast")
        return simulator

    carbon_pool_keys = (
        "metabolic_litter_c",
        "structural_litter_c",
        "fast_som_c",
        "slow_som_c",
        "passive_som_c",
    )
    vegetation_carbon_keys = ("leaf_carbon", "froot_carbon")

    with (
        caplog.at_level("WARNING"),
        warnings.catch_warnings(record=True) as captured,
    ):
        warnings.simplefilter("always")
        continuous = build_simulator()
        before = {
            key: continuous.state_manager.get_state(key).clone()
            for key in (
                *carbon_pool_keys,
                *vegetation_carbon_keys,
                "nh4_nitrogen",
                "no3_nitrogen",
            )
        }
        first_updates = continuous.step(dt=dt)
        after_first = {
            key: continuous.state_manager.get_state(key).clone()
            for key in before
        }
        checkpoint = continuous.state_manager.checkpoint()
        continuous_second = continuous.step(dt=dt)

        restarted = build_simulator()
        restarted.state_manager.restore(checkpoint)
        restarted.current_time = dt
        restarted.step_count = 1
        restarted_second = restarted.step(dt=dt)

    assert not [
        str(item.message)
        for item in captured
        if issubclass(item.category, UserWarning)
    ]
    assert not caplog.records, [record.getMessage() for record in caplog.records]

    for key in (
        *carbon_pool_keys,
        *vegetation_carbon_keys,
        "leaf_nitrogen",
        "froot_nitrogen",
        "plant_nitrogen_pool",
        "organic_nitrogen",
        "microbial_nitrogen",
        "nh4_nitrogen",
        "no3_nitrogen",
        "plant_n_storage",
        "carbon_decomposition_rate",
        "organic_nitrogen_input",
    ):
        torch.testing.assert_close(
            restarted.state_manager.get_state(key),
            continuous.state_manager.get_state(key),
            rtol=0.0,
            atol=0.0,
        )
    for key in ("nh4_nitrogen", "no3_nitrogen"):
        torch.testing.assert_close(
            restarted_second[key],
            continuous_second[key],
            rtol=0.0,
            atol=0.0,
        )

    for key in ("nh4_nitrogen", "no3_nitrogen"):
        state_delta = after_first[key] - before[key]
        torch.testing.assert_close(
            state_delta,
            first_updates[key],
            rtol=0.0,
            atol=0.0,
        )
    # Pinned after restricting lignin inhibition to structural litter
    # (2026-09-14, Parton et al. 1987 Eq. [2]-[4]).
    assert _to_float(first_updates["nh4_nitrogen"]) == pytest.approx(
        0.009922873014957395
    )
    # Re-pinned 2026-09-26: nitrate leaching reads the soil_depth forcing input of
    # this one-layer column (0.1 m) instead of the retired 1.0 m soil_depth_default.
    assert _to_float(first_updates["no3_nitrogen"]) == pytest.approx(
        -7.216657155889594e-06
    )

    litter_n_flux = (
        first_updates["leaf_litterfall_nitrogen"]
        + first_updates["froot_litterfall_nitrogen"]
    )
    nitrogen_boundary_residual = (
        first_updates["organic_nitrogen_input"] - 1000.0 * litter_n_flux
    )
    torch.testing.assert_close(
        nitrogen_boundary_residual,
        torch.zeros_like(nitrogen_boundary_residual),
        rtol=0.0,
        atol=0.0,
    )

    vegetation_carbon_delta_g = 1000.0 * torch.stack(
        [after_first[key] - before[key] for key in vegetation_carbon_keys]
    ).sum()
    pedosphere_carbon_delta = torch.stack(
        [after_first[key] - before[key] for key in carbon_pool_keys]
    ).sum()
    respiration_step = first_updates["heterotrophic_respiration"].to(
        dtype=pedosphere_carbon_delta.dtype
    ) * torch.as_tensor(dt, dtype=pedosphere_carbon_delta.dtype)
    carbon_residual = (
        vegetation_carbon_delta_g
        + pedosphere_carbon_delta
        + respiration_step
    )
    carbon_scale = torch.stack(
        (
            torch.abs(vegetation_carbon_delta_g),
            torch.abs(pedosphere_carbon_delta),
            torch.abs(respiration_step),
        )
    ).max()
    assert torch.abs(carbon_residual) <= (
        16.0 * torch.finfo(torch.float32).eps * carbon_scale
    )

    order = continuous.dependency_graph.topological_sort()
    assert order.index("static_phenology") < order.index(
        "litterfall_interface_mapper"
    )
    assert order.index("litterfall_interface_mapper") < order.index(
        "first_order_decomposition"
    )
    assert order.index("first_order_decomposition") < order.index(
        "mineralization"
    )
    assert order.index("mineralization") < order.index("nitrification")

    expected_sources = {
        "nh4_nitrogen": {
            "NitrogenCycling",
            "mineralization",
            "nitrification",
            "root_nutrient_uptake",
        },
        "no3_nitrogen": {
            "denitrification",
            "nitrification",
            "nitrogen_leaching",
            "root_nutrient_uptake",
        },
    }
    for key, expected in expected_sources.items():
        assert {
            item["functor"]
            for item in continuous.get_variable_provenance(key)
            if item.get("role") == "increment_source"
        } == expected

    assert {
        item["functor"]
        for item in continuous.get_variable_provenance(
            "carbon_decomposition_rate"
        )
    } == {"first_order_decomposition"}
    assert {
        item["functor"]
        for item in continuous.get_variable_provenance(
            "organic_nitrogen_input"
        )
    } == {"litterfall_interface_mapper"}

    registry = continuous.state_manager._contract_registry
    assert registry is not None
    expected_units = {
        "carbon_decomposition_rate": "g C m-2 s-1",
        "organic_nitrogen_input": "g N m-2 s-1",
        "nh4_nitrogen": "g N m-2",
        "no3_nitrogen": "g N m-2",
    }
    for key, unit in expected_units.items():
        contract = registry.get_contract(key)
        assert contract is not None
        assert contract.udunits == unit

    for functor_name in (
        "static_phenology",
        "litterfall_interface_mapper",
        "first_order_decomposition",
        "mineralization",
        "nitrification",
        "denitrification",
        "nitrogen_leaching",
        "root_nutrient_uptake",
        "NitrogenCycling",
    ):
        assert continuous.get_effective_device(functor_name).type == "cpu"
        assert restarted.get_effective_device(functor_name).type == "cpu"


def test_aggregated_n2o_rate_replaces_prior_timestep_value():
    """Same-step N2O writers aggregate, while the next step replaces the rate."""
    simulator = RELICSimulator(
        {
            "max_time": 20.0,
            "time_step": 10.0,
            "conservation_checks": False,
            "strict_scale_detection": False,
        }
    )
    simulator.initialize_state(
        {
            "organic_nitrogen": torch.tensor(5.0),
            "organic_nitrogen_input": torch.tensor(0.0),
            "nh4_nitrogen": torch.tensor(0.1),
            "no3_nitrogen": torch.tensor(0.2),
            "microbial_nitrogen": torch.tensor(0.05),
            "soil_temperature": torch.tensor(288.15),
            "soil_moisture": torch.tensor(0.3),
            "water_filled_pore_space": torch.tensor(0.5),
            "soil_ph": torch.tensor(6.5),
            "labile_carbon": torch.tensor(0.1),
            "carbon_decomposition_rate": torch.tensor(1.0e-6),
        }
    )
    simulator.register_functor(MineralizationProcess(), execution_group="fast")
    simulator.register_functor(NitrificationProcess(), execution_group="fast")
    simulator.register_functor(DenitrificationProcess(), execution_group="fast")

    first_updates = simulator.step(dt=10.0)
    first_rate = first_updates["n2o_flux"]
    if isinstance(first_rate, UpdateItem):
        first_rate = first_rate.value
    first_stored = simulator.state_manager.get_state("n2o_flux").clone()

    second_updates = simulator.step(dt=10.0)
    second_rate = second_updates["n2o_flux"]
    if isinstance(second_rate, UpdateItem):
        second_rate = second_rate.value
    second_stored = simulator.state_manager.get_state("n2o_flux").clone()

    assert torch.allclose(first_stored, first_rate)
    assert torch.allclose(second_stored, second_rate)
    assert not torch.allclose(second_stored, first_stored + second_rate)


if __name__ == "__main__":
    pytest.main([__file__])


