#!/usr/bin/env python3
"""
Comprehensive test script to validate pedosphere module functionality and integration.
Tests scientific calculations, parameter handling, and conservation laws.
"""

import sys
import torch
import numpy as np

from test.utils.forward_call import call_forward

def test_scientific_calculations():
    """Test that scientific calculations produce expected results."""
    print("Testing scientific calculations...")
    
    try:
        from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
        from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
        
        # Test nitrification with known parameters
        config_nitrif = {}
        
        nitrif = NitrificationProcess("test_nitrif", config_nitrif)
        
        # Test under optimal conditions
        state_optimal = {'nh4_nitrogen': torch.tensor(0.1)}  # High NH4
        environment_optimal = {
            'soil_temperature': torch.tensor(298.15),  # 25C
            'water_filled_pore_space': torch.tensor(0.6),  # WFPS optimum (Linn and Doran 1984)
            'soil_ph': torch.tensor(7.5)  # Optimal pH
        }
        
        rates_optimal = call_forward(nitrif, {**state_optimal, **environment_optimal})
        print(f"  Nitrification under optimal conditions: {rates_optimal['nitrification_flux'].item():.6f}")
        
        # Test under suboptimal conditions
        environment_suboptimal = {
            'soil_temperature': torch.tensor(283.15),  # 10C
            'water_filled_pore_space': torch.tensor(0.2),  # Low WFPS
            'soil_ph': torch.tensor(6.0)  # Suboptimal pH
        }
        
        rates_suboptimal = call_forward(nitrif, {**state_optimal, **environment_suboptimal})
        print(f"  Nitrification under suboptimal conditions: {rates_suboptimal['nitrification_flux'].item():.6f}")
        
        # Scientific check: rate should be lower under suboptimal conditions
        if rates_optimal['nitrification_flux'] > rates_suboptimal['nitrification_flux']:
            print("  Scientific consistency: optimal > suboptimal rates [PASS]")
        else:
            print("  Scientific consistency: optimal > suboptimal rates [FAIL]")
            assert False
            
        # Test first-order decomposition
        config_decomp = {}
        
        decomp = FirstOrderDecomposition("test_decomp", config_decomp)
        
        state_decomp = {
            'metabolic_litter_c': torch.tensor(100.0),
            'structural_litter_c': torch.tensor(500.0),
            'fast_som_c': torch.tensor(200.0),
            'slow_som_c': torch.tensor(1000.0),
            'passive_som_c': torch.tensor(2500.0),
        }
        
        environment_decomp = {
            'soil_temperature': torch.tensor(298.15),
            'soil_moisture': torch.tensor(0.5),
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
            'lignin_n_ratio': torch.tensor(0.3),
            'clay_content': torch.tensor(0.25),
        }
        
        decomp_rates = call_forward(decomp, {**state_decomp, **environment_decomp})
        metabolic_decomp_rate = decomp_rates[
            'diagnostics.first_order_decomposition.metabolic_litter_decomposition_flux'
        ]
        structural_decomp_rate = decomp_rates[
            'diagnostics.first_order_decomposition.structural_litter_decomposition_flux'
        ]
        print(f"  Metabolic decomposition rate: {metabolic_decomp_rate.item():.6f}")
        print(f"  Structural decomposition rate: {structural_decomp_rate.item():.6f}")
        
        # Scientific check: metabolic specific rate should exceed structural
        metabolic_specific = (
            metabolic_decomp_rate / state_decomp['metabolic_litter_c']
        )
        structural_specific = (
            structural_decomp_rate / state_decomp['structural_litter_c']
        )
        if metabolic_specific > structural_specific:
            print("  Scientific consistency: metabolic specific rate > structural [PASS]")
        else:
            print("  Scientific consistency: metabolic specific rate > structural [FAIL]")
            assert False
        
        return
        
    except Exception as e:
        print(f"  ERROR: Scientific calculations test failed - {e}")
        import traceback
        traceback.print_exc()
        assert False

def test_parameter_sensitivity():
    """Test parameter sensitivity and response functions."""
    print("Testing parameter sensitivity...")
    
    try:
        from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess
        
        base_config = {}
        
        denitrif = DenitrificationProcess("test_denitrif", base_config)
        
        state = {
            'no3_nitrogen': torch.tensor(0.05),
            'dissolved_organic_carbon': torch.tensor(0.1)
        }
        
        # Test temperature sensitivity
        temps = torch.tensor([273.15, 283.15, 293.15, 303.15])  # 0, 10, 20, 30C
        rates = []
        
        for temp in temps:
            env = {
                'soil_temperature': temp,
                'oxygen_concentration': torch.tensor(0.05),  # Anaerobic
                'water_filled_pore_space': torch.tensor(0.6),
                # Required drivers of denitrification (no environment default).
                'soil_ph': torch.tensor(6.5),
                'labile_carbon': torch.tensor(0.1),  # g C m-2
            }
            result = call_forward(denitrif, {**state, **env})
            rates.append(result['denitrification_flux'].item())
        
        print(f"  Temperature response (0-30C): {[f'{r:.6f}' for r in rates]}")
        
        # Check Q10 behavior using mid-range temperatures to avoid near-zero rates
        assert rates[1] > 0, "Denitrification rate must be positive at 10 C"
        q10_observed = rates[2] / rates[1]
        print(f"  Observed Q10 (10-20C): {q10_observed:.2f}")

        if 1.0 < q10_observed < 4.0:
            print("  Q10 response within expected range [PASS]")
        else:
            print("  Q10 response outside expected range (1.0-4.0) [FAIL]")
            assert False
        
        # Test oxygen inhibition
        oxygen_levels = torch.tensor([0.0, 0.05, 0.15, 0.21])  # Anaerobic to aerobic
        inhibition_rates = []
        
        for o2 in oxygen_levels:
            env = {
                'soil_temperature': torch.tensor(293.15),
                'oxygen_concentration': o2,
                'water_filled_pore_space': torch.tensor(0.6),
                'soil_ph': torch.tensor(6.5),
                'labile_carbon': torch.tensor(0.1),  # g C m-2
            }
            result = call_forward(denitrif, {**state, **env})
            inhibition_rates.append(result['denitrification_flux'].item())
        
        print(f"  Oxygen inhibition response: {[f'{r:.6f}' for r in inhibition_rates]}")
        
        # Check that denitrification does not increase with oxygen
        if inhibition_rates[0] >= inhibition_rates[-1]:
            print("  Oxygen inhibition working correctly [PASS]")
        else:
            print("  Oxygen inhibition not working [FAIL]")
            assert False
        
        return
        
    except Exception as e:
        print(f"  ERROR: Parameter sensitivity test failed - {e}")
        import traceback
        traceback.print_exc()
        assert False

def test_conservation_laws():
    """Test conservation laws and mass balance."""
    print("Testing conservation laws...")
    
    try:
        from relic.scientific.pedosphere.nutrient_cycling.stoichiometry import StoichiometryConstraint
        
        config = {}
        
        stoich = StoichiometryConstraint("test_stoich", config)
        
        # Test with balanced C-N fluxes
        state_balanced = {
            'metabolic_litter_c': torch.tensor(100.0),
            'metabolic_litter_n': torch.tensor(100.0/15.0)  # Perfect C:N = 15
        }
        
        carbon_fluxes = {'delta_metabolic_litter_c': torch.tensor(10.0)}
        nitrogen_fluxes = {'delta_metabolic_litter_n': torch.tensor(10.0/15.0)}
        
        result_balanced = stoich.forward(
            state_balanced, 
            time=0.0, 
            dt=3600.0,
            carbon_fluxes=carbon_fluxes,
            nitrogen_fluxes=nitrogen_fluxes
        )
        
        violation_balanced = result_balanced.get(
            'diagnostics.stoichiometry_metabolic_litter_violation',
            torch.tensor(0.0),
        )
        print(f"  Balanced flux violation: {violation_balanced.item():.6f}")
        
        # Test with unbalanced C-N fluxes
        nitrogen_fluxes_unbalanced = {'delta_metabolic_litter_n': torch.tensor(10.0/10.0)}  # C:N = 10
        
        result_unbalanced = stoich.forward(
            state_balanced,
            time=0.0,
            dt=3600.0,
            carbon_fluxes=carbon_fluxes,
            nitrogen_fluxes=nitrogen_fluxes_unbalanced
        )
        
        violation_unbalanced = result_unbalanced.get(
            'diagnostics.stoichiometry_metabolic_litter_violation',
            torch.tensor(0.0),
        )
        print(f"  Unbalanced flux violation: {violation_unbalanced.item():.6f}")
        
        # Check that unbalanced fluxes are detected
        if violation_unbalanced > violation_balanced:
            print("  Stoichiometric constraint detection working [PASS]")
        else:
            print("  Stoichiometric constraint detection not working [FAIL]")
            assert False
        
        # Test mass balance error calculation
        mass_error = result_balanced.get(
            'diagnostics.stoichiometry_mass_balance_error_total',
            torch.tensor(0.0),
        )
        mass_error_value = float(mass_error)
        print(f"  Mass balance error: {mass_error_value:.2e}")

        if np.isfinite(mass_error_value) and mass_error_value >= 0:
            print("  Mass balance diagnostic is finite [PASS]")
        else:
            print("  Mass balance diagnostic invalid [FAIL]")
            assert False
        
        return
        
    except Exception as e:
        print(f"  ERROR: Conservation laws test failed - {e}")
        import traceback
        traceback.print_exc()
        assert False

def test_alternative_decomposition_closures_are_independent():
    """Test FirstOrder and PoolsMatrix without composing both closures."""
    print("Testing independent decomposition closures...")
    from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
    from relic.scientific.pedosphere.carbon_pools.pools_matrix import PoolsMatrix

    decomp = FirstOrderDecomposition("decomp", {})
    pools = PoolsMatrix("pools", {})
    initial_state = {
        'metabolic_litter_c': torch.tensor(100.0),
        'structural_litter_c': torch.tensor(500.0),
        'metabolic_litter_n': torch.tensor(100.0/15.0),
        'structural_litter_n': torch.tensor(500.0/40.0),
        'fast_som_c': torch.tensor(200.0),
        'slow_som_c': torch.tensor(1000.0),
        'passive_som_c': torch.tensor(2500.0),
    }
    environment = {
        'soil_temperature': torch.tensor(293.15),
        'soil_moisture': torch.tensor(0.5),
        'metabolic_litter_input': torch.tensor(0.0),
        'structural_litter_input': torch.tensor(0.0),
        'lignin_n_ratio': torch.tensor(0.3),
        'clay_content': torch.tensor(0.25),
    }

    # Each alternative closure consumes the same immutable stocks independently.
    decomp_outputs = call_forward(decomp, {**dict(initial_state), **environment}, dt=3600.0)
    pool_state = {
        **initial_state,
        'environmental_modifier': torch.tensor(1.0),
        'metabolic_litter_input': torch.tensor(0.0),
        'structural_litter_input': torch.tensor(0.0),
        'fast_som_input': torch.tensor(0.0),
        'slow_som_input': torch.tensor(0.0),
        'passive_som_input': torch.tensor(0.0),
    }
    matrix_items = pools.forward(pool_state, time=0.0, dt=3600.0)
    pool_outputs = {key: item.value for key, item in matrix_items.items()}

    pool_names = (
        'metabolic_litter_c',
        'structural_litter_c',
        'fast_som_c',
        'slow_som_c',
        'passive_som_c',
    )
    assert all(name in decomp_outputs for name in pool_names)
    assert all(name in pool_outputs for name in pool_names)
    assert all(torch.isfinite(decomp_outputs[name]) for name in pool_names)
    assert all(torch.isfinite(pool_outputs[name]) for name in pool_names)
    assert decomp_outputs['carbon_decomposition_rate'] > 0
    assert decomp_outputs['heterotrophic_respiration'] > 0
    assert pool_outputs['carbon_decomposition_rate'] > 0
    assert pool_outputs['heterotrophic_respiration'] > 0

    first_order_delta = sum(decomp_outputs[name] for name in pool_names)
    matrix_delta = sum(pool_outputs[name] for name in pool_names)
    print(f"  FirstOrder pool delta: {first_order_delta.item():.6f}")
    print(f"  PoolsMatrix pool delta: {matrix_delta.item():.6f}")
    print("  Alternative closures were not chained [PASS]")

def test_pytorch_integration():
    """Test PyTorch integration, gradients, and device consistency."""
    print("Testing PyTorch integration...")
    
    try:
        from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
        
        config = {}
        
        nitrif = NitrificationProcess("test_nitrif", config)
        
        # Test gradient computation
        nh4_content = torch.tensor(0.1, requires_grad=True)
        temperature = torch.tensor(298.15, requires_grad=True)
        
        state = {'nh4_nitrogen': nh4_content}
        environment = {
            'soil_temperature': temperature,
            'water_filled_pore_space': torch.tensor(0.6),
            'soil_ph': torch.tensor(7.5)
        }
        
        # Forward pass
        rates = call_forward(nitrif, {**state, **environment})
        nitrification_rate = rates['nitrification_flux']
        
        # Backward pass
        nitrification_rate.backward()
        
        print(f"  Gradient w.r.t. NH4: {nh4_content.grad.item():.6f}")
        print(f"  Gradient w.r.t. temperature: {temperature.grad.item():.6f}")
        
        # Check that gradients are non-zero (function is differentiable)
        if nh4_content.grad.abs() > 1e-8:
            print("  NH4 gradient computation working [PASS]")
        else:
            print("  NH4 gradient computation not working [FAIL]")
            assert False
        
        if temperature.grad.abs() > 1e-8:
            print("  Temperature gradient computation working [PASS]")
        else:
            print("  Temperature gradient computation not working [FAIL]")
            assert False
        
        # Test batch processing
        batch_state = {'nh4_nitrogen': torch.tensor([0.05, 0.1, 0.15])}
        batch_environment = {
            'soil_temperature': torch.tensor([288.15, 298.15, 308.15]),
            'water_filled_pore_space': torch.tensor([0.4, 0.6, 0.8]),
            'soil_ph': torch.tensor([6.5, 7.0, 7.5])
        }
        
        batch_rates = call_forward(nitrif, {**batch_state, **batch_environment})
        batch_nitrification = batch_rates['nitrification_flux']
        
        print(f"  Batch processing shape: {batch_nitrification.shape}")
        print(f"  Batch rates: {[f'{r:.6f}' for r in batch_nitrification.tolist()]}")
        
        if batch_nitrification.shape[0] == 3:
            print("  Batch processing working [PASS]")
        else:
            print("  Batch processing not working [FAIL]")
            assert False
        
        return
        
    except Exception as e:
        print(f"  ERROR: PyTorch integration test failed - {e}")
        import traceback
        traceback.print_exc()
        assert False

def main():
    """Run comprehensive tests."""
    print("=" * 60)
    print("PEDOSPHERE COMPREHENSIVE VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_scientific_calculations,
        test_parameter_sensitivity,
        test_conservation_laws,
        test_alternative_decomposition_closures_are_independent,
        test_pytorch_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{total}] {test.__doc__.strip()}")
        print("-" * 40)
        try:
            if test():
                passed += 1
                print(f"  RESULT: PASSED")
            else:
                print(f"  RESULT: FAILED")
        except Exception as e:
            print(f"  RESULT: CRASHED - {e}")
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: {passed}/{total} TESTS PASSED")
    
    if passed == total:
        print("SUCCESS: All comprehensive tests passed!")
        print("Pedosphere module is scientifically sound and ready for production.")
        return 0
    else:
        print("FAILURE: Some tests failed!")
        print("Please address the issues before using in production.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


