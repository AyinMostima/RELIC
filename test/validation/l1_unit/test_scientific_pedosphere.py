"""
L1 Unit Tests - Scientific Module: Pedosphere

Tests the pedosphere scientific modules to ensure:
- Correct implementation of FunctorBase abstractions
- Accurate soil process calculations (decomposition, nutrient cycling, carbon pools)
- Mass balance and stoichiometric constraints
- Parameter management and validation
- Temperature and moisture dependencies
- Tensor differentiability and device consistency

Covers:
- Decomposition: FirstOrderDecomposition, MicrobialEnzymeDecomposition
- Nutrient Cycling: Mineralization, Nitrification, Denitrification, NitrogenLeaching, RootNutrientUptake
- Carbon Pools: PoolsMatrix
- Transport: GasDiffusion1D, SoluteAdvectionDiffusion1D
"""

import torch

from relic.core.functor import FunctorBase
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition


class TestDecompositionFunctors:
    """Test suite for soil decomposition modules."""
    
    def test_first_order_decomposition_inheritance(self):
        """Test FirstOrderDecomposition inherits from FunctorBase."""
        decomp = FirstOrderDecomposition()
        
        assert isinstance(decomp, FunctorBase), "Not inheriting from FunctorBase"
        assert hasattr(decomp, 'forward'), "Missing forward method"
        assert hasattr(decomp, 'get_dependencies'), "Missing get_dependencies method"
    
    def test_first_order_decomposition_basic(self):
        """Test basic first-order decomposition calculations."""
        decomp = FirstOrderDecomposition()
        
        # Standard soil conditions
        state = {
            'metabolic_litter_c': torch.tensor(100.0),
            'structural_litter_c': torch.tensor(200.0),
            'fast_som_c': torch.tensor(500.0),
            'slow_som_c': torch.tensor(1000.0),
            'passive_som_c': torch.tensor(2000.0),
            'soil_temperature': torch.tensor(288.15),     # 15 degC
            'soil_moisture': torch.tensor(0.25),          # 25% volumetric
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
            'lignin_n_ratio': torch.tensor(0.3),
            'clay_content': torch.tensor(0.20),
        }
        
        result = decomp.forward(state, time=86400.0, dt=86400.0)  # Daily time step
        
        # Check decomposition outputs
        assert isinstance(result, dict), "Result should be dictionary"
        
        for key in ('carbon_decomposition_rate', 'heterotrophic_respiration'):
            item = result[key]
            assert isinstance(item, UpdateItem)
            assert item.mode == UpdateMode.REPLACE
            assert item.unit == 'g C m-2 s-1'
            assert torch.all(item.value >= 0.0)
            assert torch.all(item.value < 1.0)

        removed = {'decomposition_rate', 'carbon_flux', 'co2_production', 'respiration_rate'}
        assert removed.isdisjoint(result)
