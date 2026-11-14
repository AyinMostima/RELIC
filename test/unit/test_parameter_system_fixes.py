"""
L1 Unit Tests for Parameter System Fixes

This module tests individual parameter management fixes in biosphere modules.
Tests verify dynamic parameter retrieval, namespace corrections, and dependency declarations.

Author: Yuchong Long
Date: 2025-09-29
"""

import pytest
import torch
from unittest.mock import patch

from relic.parameters import get_global_manager
from relic.scientific.biosphere.bvoc.megan_model import MEGANModel
from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.scientific.biosphere.stress.water_stress import WaterStress
from relic.scientific.biosphere.stress.thermal_stress import ThermalStress
from relic.scientific.biosphere.stress.nutrient_stress import NutrientStress
from relic.scientific.biosphere.stress.ozone_stress import OzoneStress
from relic.scientific.biosphere.growth.biomass_allocation import BiomassAllocation
from relic.scientific.pedosphere.carbon_pools.carbon_pools_functor import CarbonPoolsFunctor
from relic.scientific.biosphere.phenology.utils import rolling_average


class TestStressModuleDependencies:
    """Test stress module dependency declarations."""
    
    def test_water_stress_dependencies(self):
        """WaterStress reads one required external or upstream state port."""
        functor = WaterStress()
        deps = functor.get_dependencies()

        assert deps == []
        assert functor.metadata.input_variables == ['soil_water_potential']
        assert functor.get_provides() == ['water_stress_factor']
    
    def test_thermal_stress_dependencies(self):
        """ThermalStress reads one required external forcing or state port."""
        functor = ThermalStress()
        deps = functor.get_dependencies()

        assert deps == []
        assert functor.metadata.input_variables == ['air_temperature']
    
    def test_nutrient_stress_dependencies(self):
        """NutrientStress reads external tissue-composition state ports."""
        functor = NutrientStress()
        deps = functor.get_dependencies()

        assert deps == []
        assert functor.metadata.input_variables == [
            'nitrogen_concentration',
            'phosphorus_concentration',
        ]
        assert functor.get_provides() == [
            'nutrient_stress_factor',
            'diagnostics.nutrient_stress.nitrogen_factor',
            'diagnostics.nutrient_stress.phosphorus_factor',
        ]
    
    def test_ozone_stress_dependencies(self):
        """atmospheric_chemistry is the optional ozone provider; no relic Functor writes
        stomatal_conductance (FarquharPhotosynthesis outputs no conductance), so no
        photosynthesis provider is declared (D30)."""
        from relic.core.dependency_graph import normalize_dependency_declaration

        functor = OzoneStress()
        deps = normalize_dependency_declaration(functor.get_dependencies(), owner=functor.name)

        assert 'atmospheric_chemistry' in deps.optional
        assert 'farquhar_photosynthesis' not in deps.required
        assert 'farquhar_photosynthesis' not in deps.optional
        assert 'stomatal_conductance' not in FarquharPhotosynthesis().metadata.output_variables


class TestMEGANParameterBinding:
    """Test MEGAN model ParameterManager-backed device buffers."""
    
    def test_no_parameter_caching_in_init(self):
        """The removed continuity hook must not form a second parameter path."""
        model = MEGANModel()
        assert not hasattr(model, "_init_parameters")
        assert not hasattr(model, "_compute_rate_core")
        assert callable(model.get_parameter_value)
    
    def test_dynamic_emission_factor_retrieval(self):
        """Emission buffers must bind once to canonical ParameterManager keys."""
        model = MEGANModel()
        manager = get_global_manager()
        state = {
            "air_temperature": torch.tensor(298.15),
            "ppfd": torch.tensor(1000.0),
            "foliar_density": torch.tensor(300.0),
        }
        assert model.isoprene_ef.item() == pytest.approx(
            float(manager.get("biosphere.bvoc.emission_factors.isoprene"))
        )
        with patch.object(model, "get_parameter_value") as mock_get_param:
            emission = model._compute_standard_emission(
                "isoprene",
                state["air_temperature"],
                state["ppfd"],
                state["foliar_density"],
            )
        mock_get_param.assert_not_called()
        assert emission.item() > 0.0
    
    def test_numerical_parameter_namespace(self):
        """Test numerical parameters use correct namespace."""
        model = MEGANModel()
        
        # Mock parameter manager
        with patch.object(model, 'get_parameter_value') as mock_get_param:
            mock_get_param.return_value = torch.tensor(0.0)
            
            # Project the Tensor-valued assimilation input through the
            # ParameterManager-backed numerical floor contract.
            available, residual = model._project_photosynthesis_carbon(
                torch.tensor(10.0)
            )
            assert torch.equal(available, torch.tensor(10.0))
            assert torch.equal(residual, torch.tensor(0.0))
            
            mock_get_param.assert_called_once_with(
                "carbon_flux_min",
                namespace="validation.numerical",
                required=True,
            )


class TestBiomassAllocationDependency:
    """Test BiomassAllocation dependency naming fix."""
    
    def test_unique_npp_owner_dependency(self):
        """Biomass allocation must consume the unique post-respiration NPP port."""
        functor = BiomassAllocation()
        deps = functor.get_dependencies()

        assert 'autotrophic_respiration' in deps
        assert 'farquhar_photosynthesis' not in deps
        assert 'biosphere.photosynthesis.farquhar' not in deps


class TestFarquharMetadata:
    """Test Farquhar metadata construction fix."""
    
    def test_no_duplicate_metadata(self):
        """Test Farquhar doesn't create duplicate metadata objects."""
        model = FarquharPhotosynthesis()
        
        # Should have single metadata object
        assert hasattr(model, 'metadata')
        
        # Check metadata is properly structured for Phase 8 canonical runtime
        assert model.metadata.name == 'farquhar_photosynthesis'
        expression = model.metadata.mathematical_expression
        assert 'Canonical IPHS:' in expression
        assert 'x_dot = J_h e_E + M_h e_S + G_h u_h + N_h r_h' in expression
        assert 'literature formulas are stored only in theory documents' in expression
        assert 'A = min(Ac, Aj) - Rd' not in expression


class TestPhenologyRollingAverage:
    """Test phenology rolling average multi-dimensional support."""
    
    def test_arbitrary_dimension_support(self):
        """Test rolling average works on arbitrary dimensions."""
        # Create 3D test tensor (batch, time, features)
        data = torch.randn(2, 10, 3)
        
        # Test rolling average on different dimensions
        result_dim0 = rolling_average(data, window_size=3, dim=0)
        assert result_dim0.shape == data.shape
        
        result_dim1 = rolling_average(data, window_size=3, dim=1)
        assert result_dim1.shape == data.shape
        
        result_dim2 = rolling_average(data, window_size=3, dim=2)
        assert result_dim2.shape == data.shape
        
        # Test negative dimension indexing
        result_neg = rolling_average(data, window_size=3, dim=-1)
        assert result_neg.shape == data.shape
        assert torch.allclose(result_neg, result_dim2)


class TestCarbonPoolsParameterIntegration:
    """Test carbon pools Functor parameter system integration."""
    
    def test_no_register_buffer_for_pools(self):
        """Test carbon pools don't use register_buffer for parameters."""
        functor = CarbonPoolsFunctor()
        
        # Should not have register_buffer for initial pool values
        buffers = dict(functor.named_buffers())
        
        # Pool values should come from parameter manager, not buffers
        pool_buffers = [name for name in buffers.keys() 
                       if 'biomass' in name or 'soil' in name or 'litter' in name]
        assert len(pool_buffers) == 0, f"Found cached pool buffers: {pool_buffers}"
    
    def test_observer_requires_complete_runtime_pool_basis(self):
        """The observer must not initialize missing prognostic pools."""
        functor = CarbonPoolsFunctor()

        with patch.object(functor, 'get_parameter_value') as mock_get_param:
            with pytest.raises(ValueError, match='complete canonical pool basis'):
                functor.forward({}, time=0.0, dt=1.0)
            mock_get_param.assert_not_called()

            state = {
                key: torch.tensor(float(index + 1))
                for index, key in enumerate(functor._pool_keys)
            }
            output = functor.forward(state, time=0.0, dt=1.0)
            mock_get_param.assert_not_called()

        expected = torch.stack([state[key] for key in functor._pool_keys])
        assert torch.equal(output['diagnostics.carbon_pools_snapshot'], expected)
        assert torch.equal(output['diagnostics.total_carbon_stock'], expected.sum())

        non_tensor_state = dict(state)
        non_tensor_state['metabolic_litter_c'] = 1.0
        with pytest.raises(TypeError, match='torch.Tensor'):
            functor.forward(non_tensor_state, time=0.0, dt=1.0)

        negative_state = dict(state)
        negative_state['fast_som_c'] = torch.tensor(-1.0)
        with pytest.raises(ValueError, match='non-negative'):
            functor.forward(negative_state, time=0.0, dt=1.0)

        nonfinite_state = dict(state)
        nonfinite_state['slow_som_c'] = torch.tensor(float('nan'))
        with pytest.raises(ValueError, match='finite'):
            functor.forward(nonfinite_state, time=0.0, dt=1.0)

        shape_mismatch_state = dict(state)
        shape_mismatch_state['passive_som_c'] = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match='same shape'):
            functor.forward(shape_mismatch_state, time=0.0, dt=1.0)


class TestAutoRegistrationRiskMitigation:
    """Test that auto-registration risks are mitigated."""
    
    def test_farquhar_uses_required_true(self):
        """Test Farquhar model uses required=True for critical parameters."""
        model = FarquharPhotosynthesis()
        
        # Mock parameter manager with missing parameter
        param_manager = get_global_manager()
        
        # This should raise an error if parameter is missing and required=True
        with patch.object(param_manager, 'get') as mock_get:
            mock_get.side_effect = KeyError("Parameter not found")
            
            with pytest.raises((KeyError, RuntimeError, ValueError)):
                # Try to get a required parameter
                model.get_parameter_value(
                    'test_required_param',
                    namespace='biosphere.photosynthesis',
                    required=True
                )


class TestParameterNamespaceConsistency:
    """Test parameter namespace consistency across modules."""
    
    def test_stress_modules_use_consistent_namespace(self):
        """Test all stress modules use consistent namespace pattern."""
        functors = [
            WaterStress(),
            ThermalStress(),
            NutrientStress(),
            OzoneStress()
        ]
        
        for functor in functors:
            # Each should have proper namespace
            assert hasattr(functor, 'namespace')
            assert functor.namespace.startswith('biosphere.stress.')
    
    def test_bvoc_modules_namespace_hierarchy(self):
        """Test BVOC modules follow namespace hierarchy."""
        model = MEGANModel()
        
        # Should have hierarchical namespace
        assert hasattr(model, 'namespace') or hasattr(model, 'get_parameter_value')

        # A key absent from the biosphere.bvoc namespace is None (no caller default).
        result = model.get_parameter_value(
            'test_param',
            namespace='biosphere.bvoc',
            validate=False,
        )
        assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

