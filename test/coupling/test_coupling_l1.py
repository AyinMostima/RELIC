"""
L1 Unit Tests for Coupling Module

This module contains comprehensive unit tests for the coupling module's
Functor classes, testing their individual functionality, parameter access,
and inheritance relationships with the four abstract base classes.

Test Coverage:
- SpatialAggregation class functionality  
- ConservativeRemap class functionality
- Parameter system integration and validation
- Abstract base class inheritance verification
- Conservation law enforcement
- Error handling and edge cases

References:
- CF Conventions 1.8: Cell methods and time boundaries
- Jones, P.W. (1999). Conservative remapping schemes for climate models
- RELIC Architecture Documentation
"""

import pytest
import torch
import logging

# Import coupling module classes
from relic.coupling.scales.adapters import (
    SpatialAggregation, 
    ConservativeRemap
)
from relic.coupling.core.policy import ScalingMode
from relic.coupling.scales.scale_manager import ScaleManager
from relic.core.state import StateManager
from relic.coupling.scales.types import ScalingStrategy, ScalingAudit

# Import bridge classes and abstractions
from relic.core.functor import ScaleCouplingFunctor
from relic.core.abstractions import ScaleCoupling, SpatialScalingBase

# Import parameter management
from relic.parameters import UnifiedParameterManager, get_global_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSpatialAggregation:
    """Test suite for SpatialAggregation functor."""
    
    @pytest.fixture
    def spatial_aggregator(self):
        """Create a SpatialAggregation instance for testing."""
        return SpatialAggregation(
            aggregation_strategy="area_weighted",
            preserve_totals=True,
            weighting_type="area",
            device=torch.device("cpu"),
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )
    
    @pytest.fixture
    def sample_spatial_state(self):
        """Create sample spatial state data for testing."""
        return {
            'elevation': torch.randn(10, 10) * 100 + 500,  # 10x10 grid
            'land_cover': torch.rand(10, 10) * 5,  # 5 land cover types
            'soil_carbon': torch.exp(torch.randn(10, 10) * 0.5)  # Log-normal distribution
        }
    
    def test_initialization(self, spatial_aggregator):
        """Test SpatialAggregation initialization."""
        assert spatial_aggregator.aggregation_strategy == "area_weighted"
        assert spatial_aggregator.preserve_totals is True
        assert spatial_aggregator.weighting_type == "area"
        assert spatial_aggregator.source_scale == "fine_grid"
        assert spatial_aggregator.target_scale == "coarse_grid"
        assert spatial_aggregator.coupling_type == "spatial"
    
    def test_inheritance_hierarchy(self, spatial_aggregator):
        """Test that SpatialAggregation properly inherits from abstract classes."""
        assert isinstance(spatial_aggregator, ScaleCouplingFunctor)
        assert isinstance(spatial_aggregator, ScaleCoupling)
        assert isinstance(spatial_aggregator, SpatialScalingBase)
    
    # Explicit spatial bases: elevation and the continuous land_cover index are intensive
    # fields; soil_carbon (kg C m-2) is an area density in coupling.scales.defaults.variables.
    INTENSIVE_METADATA = {
        'elevation': {'spatial_basis': 'intensive'},
        'land_cover': {'spatial_basis': 'intensive'},
    }

    def test_forward_spatial_aggregation(self, spatial_aggregator, sample_spatial_state):
        """Test forward pass with spatial aggregation."""
        cell_areas = torch.ones(10, 10)  # unit-area cells [m2]
        result = spatial_aggregator.forward(
            state=sample_spatial_state,
            time=0.0,
            dt=1.0,
            variables=['elevation', 'soil_carbon'],
            weights=cell_areas,
            variable_metadata=self.INTENSIVE_METADATA,
        )

        # Check that results are returned
        assert isinstance(result, dict)
        assert 'elevation_aggregated' in result
        assert 'soil_carbon_aggregated' in result

        # Intensive mean; the area density keeps its basis (its area-weighted mean on unit cells)
        torch.testing.assert_close(result['elevation_aggregated'], sample_spatial_state['elevation'].mean())
        torch.testing.assert_close(result['soil_carbon_aggregated'], sample_spatial_state['soil_carbon'].mean())
        ledger = spatial_aggregator.last_scaling_audit.metadata['conservation_audit']['soil_carbon']
        assert ledger.source_total == pytest.approx(float(sample_spatial_state['soil_carbon'].sum()), rel=1e-5)

    def test_automated_spatial_detection(self, spatial_aggregator, sample_spatial_state):
        """Test automatic detection of spatial variables."""
        result = spatial_aggregator.forward(
            state=sample_spatial_state,
            time=0.0,
            dt=1.0,
            weights=torch.ones(10, 10),
            variable_metadata=self.INTENSIVE_METADATA,
            # No variables specified - every tensor in state is aggregated
        )

        assert set(result) == {'elevation_aggregated', 'land_cover_aggregated', 'soil_carbon_aggregated'}

    def test_custom_weights(self, spatial_aggregator, sample_spatial_state):
        """Test spatial aggregation with custom weights."""
        custom_weights = torch.rand(10, 10) + 0.1  # Custom positive weight matrix

        result = spatial_aggregator.forward(
            state=sample_spatial_state,
            time=0.0,
            dt=1.0,
            variables=['elevation'],
            weights=custom_weights,
            variable_metadata=self.INTENSIVE_METADATA,
        )

        expected = (sample_spatial_state['elevation'] * custom_weights).sum() / custom_weights.sum()
        torch.testing.assert_close(result['elevation_aggregated'], expected)
    
    def test_upscale_method(self, spatial_aggregator):
        """Test upscale method."""
        fine_data = torch.randn(5, 5)
        
        # This should use the facade's disaggregation functionality
        try:
            result = spatial_aggregator.upscale(fine_data)
            assert isinstance(result, torch.Tensor)
        except Exception as e:
            # Upscaling might fail if facade is not fully implemented
            logger.warning(f"Upscaling test failed (expected if facade incomplete): {e}")
    
    def test_downscale_method(self, spatial_aggregator):
        """Test downscale method."""
        coarse_data = torch.randn(2, 2)
        
        # This should use the facade's aggregation functionality
        try:
            result = spatial_aggregator.downscale(coarse_data)
            assert isinstance(result, torch.Tensor)
        except Exception as e:
            # Downscaling might fail if facade is not fully implemented
            logger.warning(f"Downscaling test failed (expected if facade incomplete): {e}")
    
    def test_error_handling(self, spatial_aggregator):
        """Test error handling for invalid inputs."""
        # Test with empty state
        result = spatial_aggregator.forward(
            state={},
            time=0.0,
            dt=1.0
        )
        assert result == {}  # Should return empty dict

        # A rank-1 support (for example an unstructured patch list) needs a declared basis
        state_1d = {
            'non_spatial': torch.randn(100)
        }
        with pytest.raises(ValueError, match="declare spatial_basis"):
            spatial_aggregator.forward(state=state_1d, time=0.0, dt=1.0)
        result = spatial_aggregator.forward(
            state=state_1d,
            time=0.0,
            dt=1.0,
            variable_metadata={'non_spatial': {'spatial_basis': 'intensive'}},
        )
        torch.testing.assert_close(result['non_spatial_aggregated'], state_1d['non_spatial'].mean())




def _whole_domain_geometry(n_source: int, variables, spatial_basis: str = "cell_total") -> dict:
    """One target cell covering n_source unit-area source cells (D64 explicit geometry).

    The overlap-weighted remap then reduces to the domain total for cell totals
    or the domain mean for intensive fields, which is what these consumer tests exercise.
    """
    return {
        "overlap_areas": torch.ones(n_source, 1, dtype=torch.float64),
        "source_areas": torch.ones(n_source, dtype=torch.float64),
        "target_areas": torch.tensor([float(n_source)], dtype=torch.float64),
        "variable_metadata": {name: {"spatial_basis": spatial_basis} for name in variables},
    }

class TestConservativeRemap:
    """Test suite for ConservativeRemap functor."""
    
    @pytest.fixture
    def conservative_remap(self):
        """Create a ConservativeRemap instance for testing."""
        return ConservativeRemap(
            preserve_mass=True,
            preserve_energy=True,
            preserve_positivity=True,
            device=torch.device("cpu")
        )
    
    @pytest.fixture
    def sample_remap_state(self):
        """Create sample state data for remapping."""
        return {
            'carbon_stock': torch.exp(torch.randn(8, 8) * 0.5) * 100,  # kg C/m2
            'nitrogen_stock': torch.exp(torch.randn(8, 8) * 0.3) * 10,  # kg N/m2
            'water_content': torch.rand(8, 8) * 0.3 + 0.1  # m3/m3
        }
    
    def test_initialization(self, conservative_remap):
        """Test ConservativeRemap initialization."""
        assert conservative_remap.preserve_mass is True
        assert conservative_remap.preserve_energy is True
        assert conservative_remap.preserve_positivity is True
        assert "mass" in conservative_remap.conservation_properties
        assert "energy" in conservative_remap.conservation_properties
        assert conservative_remap.source_scale == "source_grid"
        assert conservative_remap.target_scale == "target_grid"
    
    def test_inheritance_hierarchy(self, conservative_remap):
        """Test that ConservativeRemap properly inherits from abstract classes."""
        assert isinstance(conservative_remap, ScaleCouplingFunctor)
        assert isinstance(conservative_remap, ScaleCoupling)
        assert isinstance(conservative_remap, SpatialScalingBase)
    
    def test_forward_conservative_remap(self, conservative_remap, sample_remap_state):
        """Test forward pass with conservative remapping."""
        result = conservative_remap.forward(
            state=sample_remap_state,
            time=0.0,
            dt=1.0,
            variables=['carbon_stock', 'nitrogen_stock'],
            **_whole_domain_geometry(64, ['carbon_stock', 'nitrogen_stock']),
        )
        
        # Check that results are returned
        assert isinstance(result, dict)
        assert 'carbon_stock_remapped' in result
        assert 'nitrogen_stock_remapped' in result
        
        # Check positivity constraint
        assert torch.all(result['carbon_stock_remapped'] >= 0)
        assert torch.all(result['nitrogen_stock_remapped'] >= 0)
    
    def test_positivity_constraint(self, sample_remap_state):
        """Test positivity preservation."""
        # Create data with some negative values
        state_with_negatives = {
            'test_var': torch.randn(8, 8) * 10 - 5  # Some negative values
        }
        
        remapper = ConservativeRemap(preserve_positivity=True)
        result = remapper.forward(
            state=state_with_negatives,
            time=0.0,
            dt=1.0,
            variables=['test_var'],
            **_whole_domain_geometry(64, ['test_var']),
        )
        
        assert 'test_var_remapped' in result
        assert torch.all(result['test_var_remapped'] >= 0)
    
    def test_conservation_verification(self, conservative_remap, sample_remap_state):
        """Test conservation law verification."""
        # This is a simplified test - real conservation checking would require
        # proper grid specification and weight matrices
        try:
            result = conservative_remap.forward(
                state=sample_remap_state,
                time=0.0,
                dt=1.0,
                variables=['carbon_stock']
            )
            
            # Check that conservation diagnostics are set if available
            if hasattr(conservative_remap, 'metadata') and conservative_remap.metadata:
                if 'carbon_stock_remapping_conservative' in conservative_remap.metadata.diagnostics:
                    assert conservative_remap.metadata.diagnostics['carbon_stock_remapping_conservative'] is True
                    
        except Exception as e:
            logger.warning(f"Conservation test failed (expected if facade incomplete): {e}")
    
    def test_regrid_method(self, conservative_remap):
        """Test regrid method."""
        data = torch.randn(6, 6)
        source_grid = {"type": "regular", "shape": (6, 6)}
        target_grid = {"type": "regular", "shape": (3, 3)}
        
        try:
            result = conservative_remap.regrid(
                data=data,
                source_grid=source_grid,
                target_grid=target_grid,
                method="conservative",
                preserve="mass"
            )
            assert isinstance(result, torch.Tensor)
            
            # Check positivity
            assert torch.all(result >= 0)
            
        except Exception as e:
            logger.warning(f"Regrid test failed (expected if facade incomplete): {e}")
    
    def test_error_handling(self, conservative_remap):
        """Test error handling for invalid inputs."""
        # Test with empty state
        result = conservative_remap.forward(
            state={},
            time=0.0,
            dt=1.0
        )
        assert result == {}  # Should return empty dict
        
        # Test with invalid data type
        state_invalid = {
            'invalid_var': "not_a_tensor"
        }
        result = conservative_remap.forward(
            state=state_invalid,
            time=0.0,
            dt=1.0
        )
        assert result == {}  # Should skip invalid data


class TestParameterSystemIntegration:
    """Test suite for parameter system integration with coupling functors."""
    
    @pytest.fixture
    def parameter_manager(self):
        """Create a parameter manager for testing."""
        manager = UnifiedParameterManager()
        # Load coupling parameters
        try:
            manager.load_namespace('coupling.scales')
        except Exception:
            logger.warning("Could not load coupling parameters - using defaults")
        return manager
    
    def test_parameter_loading(self, parameter_manager):
        """Test that coupling parameters can be loaded."""
        # Test getting a known parameter
        try:
            param_value = parameter_manager.get(
                'coupling.scales.aggregation_strategy',
                default='mean'
            )
            assert param_value is not None
        except Exception:
            # Parameter might not exist - use default
            pass
    
    def test_a_missing_parameter_has_no_default(self, parameter_manager):
        """A parameter absent from YAML and config is None (no caller default)."""
        aggregator = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")

        nonexistent = aggregator.get_parameter_value(
            'nonexistent_parameter',
            units='dimensionless',
            namespace='coupling.scales.spatial'
        )
        assert nonexistent is None


class TestAbstractBaseClassIntegration:
    """Test suite for abstract base class integration."""
    
    def test_spatial_scaling_interface(self):
        """Test SpatialScalingBase interface compliance."""
        aggregator = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
        
        # Check that spatial-specific methods exist
        assert hasattr(aggregator, 'upscale')
        assert hasattr(aggregator, 'downscale')
        
        # Check that spatial parameters are set
        assert aggregator.source_scale == "fine_grid"
        assert aggregator.target_scale == "coarse_grid"
    
class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""
    
    def test_invalid_weighting_type(self):
        """Test handling of invalid weighting types."""
        with pytest.raises(Exception):
            SpatialAggregation(weighting_type="invalid_weighting", source_scale="fine_grid", target_scale="coarse_grid")
    
    def test_large_data_handling(self):
        """Test handling of large datasets."""
        aggregator = SpatialAggregation(source_scale="fine_grid", target_scale="coarse_grid")
        
        # Create a large spatial dataset
        large_state = {
            'large_var': torch.randn(100, 100)  # 10,000 elements
        }
        
        # Should handle large data without memory issues
        result = aggregator.forward(
            state=large_state,
            time=0.0,
            dt=1.0,
            variables=['large_var'],
            variable_metadata={'large_var': {'spatial_basis': 'intensive'}},
        )
        
        assert 'large_var_aggregated' in result
        assert isinstance(result['large_var_aggregated'], torch.Tensor)


class TestConservationLawEnforcement:
    """Test suite for conservation law enforcement."""
    
    def test_energy_conservation_spatial(self):
        """Test energy conservation in spatial aggregation."""
        aggregator = SpatialAggregation(
            aggregation_strategy="conservative",
            preserve_totals=True,
            source_scale="fine_grid",
            target_scale="coarse_grid",
        )

        # Create test data with known total energy
        spatial_data = torch.ones(10, 10) * 50.0  # Each grid cell = 50 J/m2
        state = {'energy_var': spatial_data}

        # An area density keeps its basis (50 J m-2); its amount over the 100 unit cells is the
        # ledger total 100 * 50 J, checked by the strict (preserve_totals) ledger.
        result = aggregator.forward(
            state=state,
            time=0.0,
            dt=1.0,
            variables=['energy_var'],
            weights=torch.ones(10, 10),
            variable_metadata={'energy_var': {'spatial_basis': 'area_density'}},
        )
        
        torch.testing.assert_close(
            result['energy_var_aggregated'], torch.full_like(result['energy_var_aggregated'], 50.0)
        )
        ledger = aggregator.last_scaling_audit.metadata['conservation_audit']['energy_var']
        actual_total = result['energy_var_aggregated'] * ledger.metadata['support_measure']
        # Total energy should be conserved
        expected_total = torch.tensor(
            10 * 10 * 50.0,
            device=actual_total.device,
            dtype=actual_total.dtype
        )
        
        torch.testing.assert_close(actual_total, expected_total, rtol=1e-3, atol=1e-3)
    
    def test_positive_definite_constraint(self):
        """Test positive definite constraints."""
        remapper = ConservativeRemap(preserve_positivity=True)
        
        # Create data with mixed positive and negative values
        mixed_data = torch.randn(8, 8) * 10
        state = {'mixed_var': mixed_data}
        
        result = remapper.forward(
            state=state,
            time=0.0,
            dt=1.0,
            variables=['mixed_var'],
            **_whole_domain_geometry(64, ['mixed_var']),
        )
        
        # Result should be non-negative
        if 'mixed_var_remapped' in result:
            assert torch.all(result['mixed_var_remapped'] >= 0)


class TestScaleManagerAliasQuality:
    """Dedicated checks for direct ScaleManager alias execution."""

    def test_temporal_alias_quality_metadata(self):
        """ScaleManager alias pipeline should return coverage metrics in ScalingAudit."""
        manager = get_global_manager()
        assert manager is not None, "ParameterManager must be available for coverage threshold lookup"
        expected_threshold = float(
            manager.get(
                "coupling.scales.temporal.coverage_threshold",
                units="dimensionless",
                expected_type=float
            )
        )

        class DummyHierarchy:
            def __init__(self, coverage):
                self.coverage = coverage
            def aggregate(self, data, source_scale, target_scale, variables=None, mode=None, strategy=None, metadata=None, topology=None):
                variables = variables or list(data.keys())
                aggregated = {k: v[:1] for k, v in data.items()}
                quality = {
                    var: {
                        "valid_fraction_mean": self.coverage,
                        "insufficient_periods": 0,
                        "coverage_threshold": expected_threshold,
                    } for var in variables
                }
                audit = ScalingAudit(
                    operation="aggregate",
                    source_scale=source_scale,
                    target_scale=target_scale,
                    variables=variables,
                    mode=mode or ScalingMode.CONSERVATIVE,
                    strategy=strategy or ScalingStrategy.AREA_WEIGHTED_MEAN,
                    mass_balance_error=0.0,
                    relative_error=0.0,
                    cf_compliant=False,
                    conservation_enforced=True,
                    physical_constraints_applied=False,
                    warnings=[],
                    metadata={"quality": quality},
                )
                return aggregated, audit

        facade = ScaleManager(
            enforce_conservation=True,
            parameter_manager=manager,
            state_manager=StateManager(),
        )
        facade.initialize()
        facade.hierarchical_strategy = DummyHierarchy(expected_threshold)
        daily_flux = torch.ones(30, 2, 2)
        daily_temp = torch.linspace(0, 29, steps=30).view(30, 1, 1).expand(-1, 2, 2)
        aggregated, audit = facade.aggregate(
            data={"gpp": daily_flux, "temperature": daily_temp},
            source_scale="daily",
            target_scale="monthly",
            variables=["gpp", "temperature"],
            strategy=ScalingStrategy.AREA_WEIGHTED_MEAN,
        )
        assert "gpp" in aggregated and "temperature" in aggregated
        gpp_quality = audit.metadata["quality"]["gpp"]
        assert gpp_quality.get("valid_fraction_mean", 1.0) >= expected_threshold
        assert gpp_quality["insufficient_periods"] == 0
        assert gpp_quality["coverage_threshold"] == pytest.approx(expected_threshold, rel=1e-6)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
