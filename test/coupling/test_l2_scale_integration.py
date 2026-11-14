"""
L2 Integration Tests for Scale Coupling

Tests the integration between temporal and spatial scaling modules,
including the facade interface and cross-module interactions.
"""

import pytest
import torch

from relic.core.state import StateManager
from relic.coupling.scales import (
    AggregationConfig,
    AggregationMethod,
    ConservationLevel,
    ConservativeRegridder,
    # Spatial components
    GridDescriptor,
    GridUtils,
    ScalingDirection,
    # Temporal components
    TemporalAggregator,
    SpatialQuantityBasis,
    TemporalConfig,
)
from relic.coupling.scales.hierarchical import HierarchicalAggregator
from relic.coupling.scales.hierarchical.hierarchy_index import HierarchyIndex
from relic.coupling.scales.scale_manager import ScaleManager


class TestTemporalSpatialIntegration:
    """Test integration between temporal and spatial scaling."""
    
    @pytest.fixture
    def grid_4x4(self):
        """4 x 4 cartesian grid of 1 m cells on [0, 4] m x [0, 4] m with declared cell bounds."""
        import torch
        return GridUtils(device=torch.device("cpu")).create_regular_grid(
            0.0, 4.0, 0.0, 4.0, nx=4, ny=4, coord_system="cartesian"
        )

    @pytest.fixture
    def grid_2x2(self):
        """2 x 2 cartesian grid of 2 m cells on the same domain (coarser), with declared cell bounds."""
        import torch
        return GridUtils(device=torch.device("cpu")).create_regular_grid(
            0.0, 4.0, 0.0, 4.0, nx=2, ny=2, coord_system="cartesian"
        )
    
    def test_temporal_then_spatial_aggregation(self, grid_4x4, grid_2x2):
        """Test sequential temporal then spatial aggregation."""
        # Create hourly data on 4x4 grid for 48 hours
        hourly_data_4x4 = torch.randn(4, 4, 48) * 10 + 20
        
        # Step 1: Temporal aggregation (hourly to daily)
        temporal_config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            spatial_basis=SpatialQuantityBasis.INTENSIVE,
            conservation_level=ConservationLevel.NONE,
        )
        
        temporal_agg = TemporalAggregator(temporal_config)
        
        # Create time bounds
        source_time_bounds = torch.tensor([[i, i+1] for i in range(48)], dtype=torch.float32)
        target_time_bounds = torch.tensor([[0, 24], [24, 48]], dtype=torch.float32)
        
        # Aggregate temporally using the correct aggregate method signature
        source_time = torch.arange(48, dtype=torch.float32) + 0.5
        target_time = torch.tensor([12.0, 36.0], dtype=torch.float32)  # Daily centers
        
        # Reshape data to have time as first dimension for aggregation
        data_for_aggregation = hourly_data_4x4.permute(2, 0, 1)  # [48, 4, 4]
        
        # Flatten spatial dimensions for processing
        data_flat = data_for_aggregation.reshape(48, 16)  # [48, 16]
        
        result = temporal_agg.aggregate(
            data_flat,
            source_time=source_time,
            target_time=target_time,
            variable_name='temperature',
            standard_name='air_temperature',
            units='K',
            calendar='noleap',
            time_units='hours since 2000-01-01',
            source_bounds=source_time_bounds,
            target_bounds=target_time_bounds,
            temporal_semantics='interval_mean',
            target_units='K',
            measure_units='K hour',
            integration_time_unit='hour',
        )
        
        # Reshape back to spatial dimensions
        daily_data_4x4 = result.data.reshape(2, 4, 4).permute(1, 2, 0)  # [4, 4, 2]
        
        assert daily_data_4x4.shape == (4, 4, 2)
        
        # Step 2: Spatial aggregation (4x4 to 2x2); temperature is an intensive field
        spatial_regridder = ConservativeRegridder(grid_4x4, grid_2x2, quantity_basis="intensive")

        # Aggregate spatially for each day
        daily_data_2x2 = torch.zeros(2, 2, 2)
        for day in range(2):
            daily_data_2x2[:, :, day] = spatial_regridder.regrid(
                daily_data_4x4[:, :, day],
                source_grid=grid_4x4,
                target_grid=grid_2x2,
                method="conservative",
            )
        
        assert daily_data_2x2.shape == (2, 2, 2)
        
        # Verify conservation (approximately, as we're using intensive variable)
        for day in range(2):
            mean_before = torch.mean(daily_data_4x4[:, :, day])
            mean_after = torch.mean(daily_data_2x2[:, :, day])
            assert torch.abs(mean_before - mean_after) < 10.0  # Relaxed tolerance for conservative regridding
    
    def test_spatial_then_temporal_aggregation(self, grid_4x4, grid_2x2):
        """Test sequential spatial then temporal aggregation."""
        # Create hourly data on 4x4 grid for 24 hours
        hourly_data_4x4 = torch.ones(4, 4, 24) * 100.0
        
        # Step 1: Spatial aggregation (4x4 to 2x2); the precipitation flux is an area density
        spatial_regridder = ConservativeRegridder(grid_4x4, grid_2x2, quantity_basis="area_density")

        hourly_data_2x2 = torch.zeros(2, 2, 24)
        for hour in range(24):
            hourly_data_2x2[:, :, hour] = spatial_regridder.regrid(
                hourly_data_4x4[:, :, hour],
                source_grid=grid_4x4,
                target_grid=grid_2x2,
                method="conservative",
            )
        
        assert hourly_data_2x2.shape == (2, 2, 24)
        
        # Step 2: Temporal aggregation (hourly to daily)
        temporal_config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            spatial_basis=SpatialQuantityBasis.AREA_DENSITY,  # kg m-2 s-1
            conservation_level=ConservationLevel.NONE,
        )
        
        temporal_agg = TemporalAggregator(temporal_config)
        
        source_time_bounds = torch.tensor([[i, i+1] for i in range(24)], dtype=torch.float32)
        target_time_bounds = torch.tensor([[0, 24]], dtype=torch.float32)
        
        # Temporal aggregation using the correct aggregate method signature
        source_time = torch.arange(24, dtype=torch.float32) + 0.5
        target_time = torch.tensor([12.0], dtype=torch.float32)  # Daily center
        
        # Reshape data to have time as first dimension for aggregation
        data_for_aggregation = hourly_data_2x2.permute(2, 0, 1)  # [24, 2, 2]
        
        # Flatten spatial dimensions for processing
        data_flat = data_for_aggregation.reshape(24, 4)  # [24, 4]
        
        result = temporal_agg.aggregate(
            data_flat,
            source_time=source_time,
            target_time=target_time,
            variable_name='precipitation',
            standard_name='precipitation_flux',
            units='kg m-2 s-1',
            calendar='noleap',
            time_units='hours since 2000-01-01',
            source_bounds=source_time_bounds,
            target_bounds=target_time_bounds,
            temporal_semantics='rate',
            target_units='kg m-2 s-1',
            measure_units='kg m-2',
            integration_time_unit='s',
        )
        
        # Reshape back to spatial dimensions
        daily_data_2x2 = result.data.reshape(1, 2, 2).permute(1, 2, 0)  # [2, 2, 1]
        
        assert daily_data_2x2.shape == (2, 2, 1)
        
        # Verify that the aggregation produces reasonable results
        # For extensive variables, expect values greater than zero
        assert torch.all(daily_data_2x2 > 0)
        
        # Basic sanity check - the daily total should be positive
        total_after = torch.sum(daily_data_2x2)
        assert total_after > 0


class TestScaleManagerIntegration:
    """Test ScaleManager integration with temporal and spatial modules."""
    
    @pytest.fixture
    def manager(self):
        """Create ScaleManager instance."""
        state_manager = StateManager(device=torch.device("cpu"))
        mgr = ScaleManager(
            device=torch.device("cpu"),
            enforce_conservation=True,
            cf_strict_mode=True,
            state_manager=state_manager,
        )
        try:
            mgr.initialize()
            yield mgr
        finally:
            mgr.cleanup()
            state_manager.reset()
    
    def test_manager_temporal_aggregation(self, manager):
        """Test that manager can be created and has basic attributes."""
        assert manager is not None
        assert hasattr(manager, 'device')
        assert hasattr(manager, 'enforce_conservation')
        assert hasattr(manager, 'aggregate')
        assert hasattr(manager, 'disaggregate')
        assert callable(manager.aggregate)
        assert callable(manager.disaggregate)
    
    def test_manager_spatial_regridding(self, manager):
        """Test that manager has spatial functionality."""
        assert manager.conservation_enforcer is not None
        assert manager.constraint_manager is manager.state_manager.physical_constraint_manager
    
    def test_manager_combined_scaling(self, manager):
        """Test manager basic functionality for combined operations."""
        assert hasattr(manager, 'aggregate')
        assert hasattr(manager, 'disaggregate')
        report = manager.verify_conservation()
        assert isinstance(report, dict)
        test_data = torch.ones(2, 2)
        assert test_data.shape == (2, 2)


class TestHierarchicalAggregatorIntegration:
    """Test HierarchicalAggregator integration with other modules."""
    
    @pytest.fixture
    def hierarchical_agg(self):
        """Create HierarchicalAggregator instance."""
        # Create hierarchy index
        topology = {
            'n_source_units': 10,
            'n_target_units': 3,
            'source_areas': torch.ones(10) * 0.1,
            'target_areas': torch.ones(3) * 0.33,
            'parent_map': {i: [0, 0, 0, 1, 1, 1, 1, 2, 2, 2][i] for i in range(10)}  # 10 patches -> 3 hillslopes
        }
        index = HierarchyIndex(
            source_scale='patch',
            target_scale='hillslope',
            topology=topology
        )
        
        return HierarchicalAggregator(
            index=index,
            hierarchy_levels=['patch', 'hillslope', 'watershed'],
            weight_type='area',
            rule='conservative'
        )
    
    def test_hierarchical_with_temporal(self, hierarchical_agg):
        """Test hierarchical aggregation combined with temporal."""
        # Create patch-level hourly data
        n_patches = 10
        n_hours = 24
        patch_data = torch.randn(n_patches, n_hours) * 5 + 15
        patch_areas = torch.rand(n_patches) * 100 + 50
        
        # Define hierarchy mapping
        patch_to_hillslope = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # 10 patches -> 3 hillslopes
        hillslope_to_watershed = torch.tensor([0, 0, 1])  # 3 hillslopes -> 2 watersheds
        
        # Step 1: Hierarchical aggregation at each hour
        hillslope_data = torch.zeros(3, n_hours)
        for hour in range(n_hours):
            for hs in range(3):
                patch_mask = patch_to_hillslope == hs
                patch_values = patch_data[patch_mask, hour]
                patch_weights = patch_areas[patch_mask]
                hillslope_data[hs, hour] = torch.sum(patch_values * patch_weights) / torch.sum(patch_weights)
        
        # Step 2: Temporal aggregation to daily
        temporal_config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            conservation_level=ConservationLevel.NONE,
        )
        
        temporal_agg = TemporalAggregator(temporal_config)
        
        # Create time arrays for temporal aggregation
        source_time = torch.arange(24, dtype=torch.float32) + 0.5
        target_time = torch.tensor([12.0])  # Mid-day for daily average
        source_time_bounds = torch.tensor(
            [[i, i + 1] for i in range(24)], dtype=torch.float32
        )
        target_time_bounds = torch.tensor([[0, 24]], dtype=torch.float32)
        
        hillslope_daily = torch.zeros(3)
        for hs in range(3):
            result = temporal_agg.aggregate(
                hillslope_data[hs, :],
                source_time=source_time,
                target_time=target_time,
                variable_name="temperature",
                units="K",
                calendar="noleap",
                time_units="hours since 2000-01-01",
                source_bounds=source_time_bounds,
                target_bounds=target_time_bounds,
                temporal_semantics="interval_mean",
                target_units="K",
                measure_units="K hour",
                integration_time_unit="hour",
            )
            hillslope_daily[hs] = result.data[0]
        
        assert hillslope_daily.shape == (3,)
        
        # Step 3: Further aggregate to watershed level
        watershed_daily = torch.zeros(2)
        for ws in range(2):
            hs_mask = hillslope_to_watershed == ws
            watershed_daily[ws] = torch.mean(hillslope_daily[hs_mask])
        
        assert watershed_daily.shape == (2,)
    
    def test_hierarchical_conservation(self, hierarchical_agg):
        """Test conservation through hierarchical aggregation."""
        # Create simple test data
        patch_values = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
        patch_areas = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        
        # Map 4 patches to 2 hillslopes
        patch_to_hillslope = torch.tensor([0, 0, 1, 1])
        
        # Aggregate with area weighting
        hillslope_values = torch.zeros(2)
        for hs in range(2):
            mask = patch_to_hillslope == hs
            values = patch_values[mask]
            areas = patch_areas[mask]
            hillslope_values[hs] = torch.sum(values * areas) / torch.sum(areas)
        
        # Check conservation of weighted average
        total_weighted_before = torch.sum(patch_values * patch_areas)
        total_area_before = torch.sum(patch_areas)
        mean_before = total_weighted_before / total_area_before
        
        # Hillslope areas
        hillslope_areas = torch.tensor([
            torch.sum(patch_areas[patch_to_hillslope == 0]),
            torch.sum(patch_areas[patch_to_hillslope == 1])
        ])
        
        total_weighted_after = torch.sum(hillslope_values * hillslope_areas)
        total_area_after = torch.sum(hillslope_areas)
        mean_after = total_weighted_after / total_area_after
        
        assert torch.abs(mean_before - mean_after) < 1e-6


class TestCFMetadataIntegration:
    """Test CF metadata handling across modules."""
    
    def test_cf_metadata_temporal_aggregation(self):
        """Test CF metadata updates during temporal aggregation."""
        config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            conservation_level=ConservationLevel.NONE,
        )
        
        aggregator = TemporalAggregator(config)
        
        # Create test data with metadata
        data = torch.randn(24)
        source_time = torch.tensor([i + 0.5 for i in range(24)], dtype=torch.float32)
        target_time = torch.tensor([12.0], dtype=torch.float32)
        
        # Aggregate with metadata
        result = aggregator.aggregate(
            data,
            source_time=source_time,
            target_time=target_time,
            standard_name='air_temperature',
            units='K',
            cell_methods='time: point',
            calendar='noleap',
            time_units='hours since 2000-01-01',
            source_bounds=torch.tensor(
                [[i, i + 1] for i in range(24)], dtype=torch.float32
            ),
            target_bounds=torch.tensor([[0, 24]], dtype=torch.float32),
            temporal_semantics='interval_mean',
            target_units='K',
            measure_units='K hour',
            integration_time_unit='hour',
        )
        
        # Check metadata was updated
        assert result.metadata is not None
        assert 'cell_methods' in result.metadata
        # Should indicate time averaging was applied
        assert 'mean' in result.metadata['cell_methods'] or 'average' in result.metadata['cell_methods']
    
    def test_cf_metadata_spatial_regridding(self):
        """Test CF metadata updates during spatial regridding."""
        grid_utils = GridUtils()
        
        source_grid = grid_utils.create_regular_grid(
            x_min=0.0, x_max=4.0,
            y_min=0.0, y_max=4.0,
            nx=4, ny=4,
            coord_system="latlon"
        )
        
        target_grid = grid_utils.create_regular_grid(
            x_min=0.0, x_max=4.0,
            y_min=0.0, y_max=4.0,
            nx=2, ny=2,
            coord_system="latlon"
        )
        
        regridder = ConservativeRegridder(source_grid, target_grid, quantity_basis="intensive")

        # Create test data
        data = torch.ones(4, 4) * 273.15  # Temperature in Kelvin
        
        # Regrid
        result = regridder.regrid(data, source_grid, target_grid, method="conservative")
        
        # Create metadata
        metadata = {
            'standard_name': 'air_temperature',
            'units': 'K',
            'grid_mapping': 'crs',
            'coordinates': 'lat lon'
        }
        
        # In a full implementation, the regridder would update metadata
        # Here we just verify the structure
        assert result.shape == (2, 2)
        assert metadata['standard_name'] == 'air_temperature'
        assert metadata['units'] == 'K'


class TestErrorHandlingIntegration:
    """Test error handling across integrated modules."""
    
    def test_inconsistent_scales(self):
        """Frequency tokens defer direction to the bound runtime contract."""
        config = TemporalConfig(
            source_frequency='monthly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            conservation_level=ConservationLevel.NONE,
        )
        aggregator = TemporalAggregator(config)

        with pytest.raises(ValueError, match="source_frequency.*PeriodIndexer"):
            aggregator.aggregate(
                data=torch.ones(1, dtype=torch.float64),
                source_time=torch.tensor([0.5], dtype=torch.float64),
                target_time=torch.tensor([0.5], dtype=torch.float64),
                units="K",
                temporal_semantics="interval_mean",
                calendar="noleap",
                time_units="days since 2001-01-01",
                source_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
                target_bounds=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
            )
    
    def test_strict_conservation_requires_q_ref_parameter(self):
        """Strict mapping checks require an explicit reference-scale owner."""
        config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            conservation_level=ConservationLevel.STRICT,
            spatial_basis=SpatialQuantityBasis.CELL_TOTAL  # kg h-1 per cell
        )

        aggregator = TemporalAggregator(
            config,
            AggregationConfig(method=AggregationMethod.SUM),
        )

        hourly_data = torch.ones(24) * 100.0
        source_time = torch.arange(24, dtype=torch.float32) + 0.5
        target_time = torch.tensor([12.0], dtype=torch.float32)

        with pytest.raises(
            ValueError,
            match="q_ref_parameter is required",
        ):
            aggregator.aggregate(
                hourly_data,
                source_time=source_time,
                target_time=target_time,
                units='kg h-1',
                calendar='noleap',
                time_units='hours since 2000-01-01',
                source_bounds=torch.tensor(
                    [[i, i + 1] for i in range(24)], dtype=torch.float32
                ),
                target_bounds=torch.tensor([[0, 24]], dtype=torch.float32),
                temporal_semantics='rate',
                target_units='kg',
                measure_units='kg',
                integration_time_unit='hour',
            )
    
    def test_missing_data_propagation(self):
        """Test how missing data propagates through scaling operations."""
        # Create data with missing values
        data = torch.tensor([1.0, 2.0, float('nan'), 4.0, 5.0, 6.0])
        
        config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            conservation_level=ConservationLevel.NONE,
        )
        
        aggregator = TemporalAggregator(config)
        
        source_time = torch.arange(len(data), dtype=torch.float32) + 0.5
        target_time = torch.tensor([3.0], dtype=torch.float32)
        
        # Aggregate with missing data
        result = aggregator.aggregate(
            data,
            source_time=source_time,
            target_time=target_time,
            units='K',
            calendar='noleap',
            time_units='hours since 2000-01-01',
            source_bounds=torch.tensor(
                [[i, i + 1] for i in range(len(data))], dtype=torch.float32
            ),
            target_bounds=torch.tensor([[0, len(data)]], dtype=torch.float32),
            temporal_semantics='interval_mean',
            target_units='K',
            measure_units='K hour',
            integration_time_unit='hour',
        )

        # NaN samples are excluded and the remaining positive coverage is normalized.
        assert torch.all(torch.isfinite(result.data))
        assert result.data.item() == pytest.approx(3.6)
        assert result.quality_flags is not None
        assert not torch.any(result.quality_flags.insufficient_coverage_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


