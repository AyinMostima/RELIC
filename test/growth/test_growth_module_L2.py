"""
L2 Integration Tests for Growth Module

This module provides comprehensive L2 integration testing for the growth module:
- Tests interaction between BiomassAllocation, PhenologyGrowth, and StructuralGrowth
- Verifies data flow and state management
- Tests parameter propagation through the system
- Validates conservation across module boundaries
"""

import pytest
import torch
import numpy as np

from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.biosphere.growth.phenology_growth import PhenologyGrowth
from relic.scientific.biosphere.growth.structural_growth import StructuralGrowth
from relic.parameters import get_global_manager
from relic.core.state import UpdateItem, UpdateMode


def _phenology_values(
    functor: PhenologyGrowth,
    state: dict[str, torch.Tensor],
    *,
    time: float = 0.0,
    dt: float = 3600.0,
) -> dict[str, torch.Tensor]:
    """Run the selected-owner activity observer and unwrap its diagnostic."""
    result = functor.forward(state, time=time, dt=dt)
    assert set(result) == set(functor.metadata.output_variables)
    assert 'rate' not in result
    assert all(isinstance(item, UpdateItem) for item in result.values())
    assert all(item.mode is UpdateMode.REPLACE for item in result.values())
    assert all(item.per_timestep is False for item in result.values())
    assert all(item.unit == 'dimensionless' for item in result.values())
    assert all(item.source_functor == functor.name for item in result.values())
    return {key: item.value for key, item in result.items()}


class TestGrowthModuleIntegration:
    """L2 Integration tests for complete growth module."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup integrated test environment."""
        self.device = torch.device('cpu')
        
        # Create all functors
        self.phenology = PhenologyGrowth(
            device=self.device,
            phenology_model_owner="gsi_phenology",
        )

        self.allocation = BiomassAllocation(name="allocation", device=self.device)

        self.structural = StructuralGrowth(name="structural", device=self.device)
        
        # Initialize state
        self.initial_state = {
            'phenological_factor': torch.tensor(0.6, device=self.device),
            'air_temperature': torch.tensor(298.15, device=self.device),
            'net_primary_productivity': torch.tensor(5.0e-6, device=self.device),
            'water_stress_factor': torch.tensor(0.8, device=self.device),
            'leaf_area_index': torch.tensor(3.0, device=self.device),
            'phenology_phase': torch.tensor(2.0, device=self.device),
            'leaf_biomass_c': torch.tensor(0.1, device=self.device),
            'wood_biomass_c': torch.tensor(0.2, device=self.device),
            'root_biomass_c': torch.tensor(0.15, device=self.device),
        }
        
    def test_phenology_to_allocation_flow(self):
        """The observer output does not become a hidden allocation fallback."""
        environment = {}
        phenology_output = _phenology_values(self.phenology, self.initial_state)

        state_with_phenology = {**self.initial_state, **phenology_output}
        allocation_output = self.allocation.forward(
            state_with_phenology,
            time=0.0,
            dt=3600.0,
        )

        assert set(phenology_output) == {
            'diagnostics.phenology.growth_active'
        }
        assert 'growth_rate_leaf' in allocation_output
        assert 'growth_rate_stem' in allocation_output
        assert 'growth_rate_root' in allocation_output

        assert bool(
            phenology_output['diagnostics.phenology.growth_active'].item()
        )

        total_growth = (
            allocation_output['growth_rate_leaf'].value
            + allocation_output['growth_rate_stem'].value
            + allocation_output['growth_rate_root'].value
        )
        npp = self.initial_state['net_primary_productivity'].to(
            dtype=total_growth.dtype,
            device=total_growth.device,
        )
        assert torch.allclose(total_growth, npp, atol=1.0e-12)
        assert 'phenological_factor' not in self.allocation.get_required_inputs()

    def test_allocation_to_structural_flow(self):
        """Test data flow from allocation to structural growth."""
        environment = {}
        phenology_output = _phenology_values(self.phenology, self.initial_state)

        state_with_phenology = {**self.initial_state, **phenology_output}
        allocation_output = self.allocation.forward(
            state_with_phenology,
            time=0.0,
            dt=3600.0,
        )

        structural_state = {
            'growth_rate_leaf': allocation_output['growth_rate_leaf'].value,
            'leaf_area_index': self.initial_state['leaf_area_index'],
        }
        structural_output = self.structural.forward(
            structural_state,
            time=0.0,
            dt=3600.0,
        )

        assert set(structural_output) == set(self.structural.metadata.output_variables)
        assert all(isinstance(item, UpdateItem) for item in structural_output.values())
        leaf_growth = allocation_output['growth_rate_leaf'].value
        lai_change = structural_output[
            'diagnostics.structural_growth.net_leaf_area_tendency'
        ].value
        if leaf_growth > 0:
            assert lai_change >= 0
            
    def test_full_growth_cycle(self):
        """Test complete growth cycle through all modules."""
        environment = {}
        states = []
        current_state = self.initial_state.copy()
        dt = 3600.0

        for timestep in range(10):
            current_state['phenological_factor'] = torch.tensor(
                (timestep + 1) / 10.0, device=self.device
            )

            pheno_output = _phenology_values(
                self.phenology,
                current_state,
                time=timestep * dt,
                dt=dt,
            )
            current_state.update(pheno_output)

            alloc_output = self.allocation.forward(
                current_state,
                time=timestep * dt,
                dt=dt,
            )
            struct_output = self.structural.forward(
                {
                    'growth_rate_leaf': alloc_output['growth_rate_leaf'].value,
                    'leaf_area_index': current_state['leaf_area_index'],
                },
                time=timestep * dt,
                dt=dt,
            )

            for key in ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c'):
                update = alloc_output[key]
                assert update.mode is UpdateMode.ACCUMULATE
                assert update.per_timestep is True
                current_state[key] = current_state[key] + update.value

            states.append({
                'timestep': timestep,
                'pheno_factor': current_state['phenological_factor'].item(),
                'growth_active': bool(
                    pheno_output[
                        'diagnostics.phenology.growth_active'
                    ].item()
                ),
                'leaf_biomass_c': current_state['leaf_biomass_c'].item(),
                'lai_tendency': struct_output[
                    'diagnostics.structural_growth.net_leaf_area_tendency'
                ].value.item(),
            })

        assert states[-1]['leaf_biomass_c'] > states[0]['leaf_biomass_c']
        assert states[-1]['pheno_factor'] != states[0]['pheno_factor']

    def test_carbon_conservation(self):
        """Test carbon conservation across all modules."""
        environment = {}
        initial_carbon = (
            self.initial_state['leaf_biomass_c']
            + self.initial_state['wood_biomass_c']
            + self.initial_state['root_biomass_c']
        )

        pheno_output = _phenology_values(self.phenology, self.initial_state)
        state_with_pheno = {**self.initial_state, **pheno_output}

        dt = 3600.0
        alloc_output = self.allocation.forward(
            state_with_pheno,
            time=0.0,
            dt=dt,
        )

        final_carbon = initial_carbon.clone()
        for key in ('leaf_biomass_c', 'wood_biomass_c', 'root_biomass_c'):
            final_carbon = final_carbon + alloc_output[key].value
        actual_gain = final_carbon - initial_carbon
        expected_gain = self.initial_state['net_primary_productivity'].to(
            dtype=actual_gain.dtype,
            device=actual_gain.device,
        ) * dt
        residual = alloc_output[
            'diagnostics.biomass_allocation.carbon_balance_residual'
        ].value
        assert torch.allclose(actual_gain, expected_gain, atol=1.0e-8)
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-12)
        
    def test_parameter_propagation(self):
        """Test that the canonical SLA ParameterSpec enters the LAI equation."""
        param_manager = get_global_manager()
        environment = {}
        pheno_output = _phenology_values(self.phenology, self.initial_state)
        state = {**self.initial_state, **pheno_output}

        alloc_output = self.allocation.forward(state, time=0.0, dt=3600.0)
        leaf_growth = alloc_output['growth_rate_leaf'].value
        output = self.structural.forward(
            {
                'growth_rate_leaf': leaf_growth,
                'leaf_area_index': state['leaf_area_index'],
            },
            time=0.0,
            dt=3600.0,
        )
        sla_spec = param_manager.get_parameter(
            'biosphere.phenology.morphology.sla'
        )
        assert sla_spec is not None
        gross = output[
            'diagnostics.structural_growth.gross_leaf_area_growth_tendency'
        ].value
        expected = leaf_growth * torch.as_tensor(
            sla_spec.value,
            dtype=leaf_growth.dtype,
            device=leaf_growth.device,
        )
        assert torch.allclose(gross, expected, atol=1.0e-12)
        
    def test_stress_response_integration(self):
        """Test integrated stress response across modules."""
        environment = {}
        
        # Test with different stress levels
        stress_levels = [1.0, 0.5, 0.1]
        results = []
        
        for stress in stress_levels:
            state = self.initial_state.copy()
            state['water_stress_factor'] = torch.tensor(stress, device=self.device)
            state['net_primary_productivity'] = self.initial_state[
                'net_primary_productivity'
            ] * stress

            pheno_output = _phenology_values(self.phenology, state)
            state.update(pheno_output)

            alloc_output = self.allocation.forward(
                state,
                time=0.0,
                dt=3600.0,
            )

            results.append({
                'stress': stress,
                'leaf_growth': alloc_output['growth_rate_leaf'].value.item(),
                'root_growth': alloc_output['growth_rate_root'].value.item(),
                'total_growth': alloc_output[
                    'diagnostics.biomass_allocation.total_growth_rate'
                ].value.item(),
                'leaf_fraction': alloc_output[
                    'diagnostics.biomass_allocation.allocation_fraction_leaf'
                ].value.item(),
                'root_fraction': alloc_output[
                    'diagnostics.biomass_allocation.allocation_fraction_root'
                ].value.item(),
            })
            
        # Under severe water stress, root allocation share should increase.
        assert results[0]['root_fraction'] < results[2]['root_fraction']
        assert results[0]['leaf_fraction'] > results[2]['leaf_fraction']
        
        # Total growth should decrease when upstream NPP is stress-limited.
        assert results[0]['total_growth'] > results[2]['total_growth']
        
    def test_seasonal_cycle(self):
        """Test allocation under an explicit selected-owner seasonal control."""
        environment = {}
        
        factor_annual_cycle = [
            0.0, 0.05, 0.3, 0.7, 1.0, 0.9,
            0.8, 0.6, 0.3, 0.1, 0.05, 0.0,
        ]
        phase_annual_cycle = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0]
        
        results = []
        current_state = self.initial_state.copy()
        
        for month, (factor, phase) in enumerate(
            zip(factor_annual_cycle, phase_annual_cycle)
        ):
            current_state['phenological_factor'] = torch.tensor(
                factor,
                device=self.device,
            )
            current_state['phenology_phase'] = torch.tensor(
                float(phase), device=self.device
            )

            pheno_output = _phenology_values(
                self.phenology,
                current_state,
                time=month * 30.0 * 86400.0,
                dt=3600.0,
            )
            current_state.update(pheno_output)
            current_state['net_primary_productivity'] = (
                self.initial_state['net_primary_productivity']
                * current_state['phenological_factor']
            )
            alloc_output = self.allocation.forward(
                current_state,
                time=month * 30.0 * 86400.0,
                dt=3600.0,
            )

            results.append({
                'month': month,
                'pheno_factor': factor,
                'growth_active': bool(
                    pheno_output[
                        'diagnostics.phenology.growth_active'
                    ].item()
                ),
                'leaf_growth': alloc_output['growth_rate_leaf'].value.item()
            })
            
        # Verify seasonal patterns
        # Growth should be highest in growing season
        growing_season = [r for r in results if r['pheno_factor'] >= 0.3]
        dormant_season = [r for r in results if r['pheno_factor'] <= 0.05]
        
        avg_growing = np.mean([r['leaf_growth'] for r in growing_season])
        avg_dormant = np.mean([r['leaf_growth'] for r in dormant_season]) if dormant_season else 0
        
        assert avg_growing > avg_dormant


def test_error_handling():
    """Test error handling across integrated modules."""
    device = torch.device('cpu')
    
    # Test with missing required inputs
    phenology = PhenologyGrowth(
        device=device,
        phenology_model_owner="gsi_phenology",
    )
    
    with pytest.raises(KeyError, match='phenological_factor'):
        phenology.forward({}, time=0.0, dt=3600.0)
    

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

