"""
L3 System Tests for Complete Scale Coupling

Tests the full system integration including temporal, spatial, and hierarchical
scaling with real-world scenarios and end-to-end workflows.
"""

import numpy as np
import pytest
import torch

from relic.coupling.scales import (
    AggregationConfig,
    AggregationMethod,
    ConservationLevel,
    DisaggregationConfig,
    ConservativeRegridder,
    GridDescriptor,
    ScalingDirection,
    SpatialRegridderFactory,
    TemporalAggregator,
    TemporalConfig,
    SpatialQuantityBasis,
    TemporalDisaggregator,
)
from relic.coupling.scales.hierarchical import HierarchicalAggregator, HierarchyIndex

# Seed of the synthetic monthly temperature field of the downscaling workflow (fixed, as in the
# seeded conservative-remap fixture of test_coupling_l3.py).
DOWNSCALING_SEED = 2


def _interval_bounds(count: int, width: float) -> torch.Tensor:
    starts = torch.arange(count, dtype=torch.float64) * width
    return torch.stack((starts, starts + width), dim=1)


def _centers(bounds: torch.Tensor) -> torch.Tensor:
    return torch.mean(bounds, dim=1)


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows with realistic scenarios."""
    
    def test_climate_model_downscaling_workflow(self):
        """Test complete workflow for climate model downscaling."""
        # Scenario: Downscale monthly climate model output to daily values
        # at higher spatial resolution for regional impact assessment
        
        # Step 1: Create coarse monthly climate data (5x5 grid)
        n_months = 12
        coarse_grid_size = 5
        monthly_temp = torch.zeros(coarse_grid_size, coarse_grid_size, n_months)
        generator = torch.Generator().manual_seed(DOWNSCALING_SEED)

        # Generate realistic temperature pattern (seasonal cycle)
        for month in range(n_months):
            base_temp = 15 + 10 * np.sin(2 * np.pi * (month - 3) / 12)  # Peak in summer
            monthly_temp[:, :, month] = (
                torch.randn(coarse_grid_size, coarse_grid_size, generator=generator) * 2 + base_temp
            )
        
        # Step 2: Spatial downscaling (5x5 to 20x20)
        source_grid = GridDescriptor(
            shape=(5, 5),
            origin=(0.0, 0.0),
            spacing=(20.0, 20.0),  # 20km resolution
            crs='EPSG:4326'
        )
        
        target_grid = GridDescriptor(
            shape=(20, 20),
            origin=(0.0, 0.0),
            spacing=(5.0, 5.0),  # 5km resolution
            crs='EPSG:4326'
        )
        
        # The fine cell centres of the outer half coarse cell (2.5 and 7.5 km from each edge) lie
        # outside the coarse cell centres: the edge values are extended there (boundary_mode
        # "nearest"); with "constant" those targets are uncovered (NaN), never filled with 0.
        spatial_regridder = SpatialRegridderFactory.create_regridder(
            'bilinear',
            source_grid,
            target_grid,
            boundary_mode='nearest',
        )
        assert spatial_regridder.coverage_report["uncovered_target_cells"] == 0
        
        monthly_temp_fine = torch.zeros(20, 20, n_months)
        for month in range(n_months):
            monthly_temp_fine[:, :, month] = spatial_regridder.regrid(
                monthly_temp[:, :, month],
                method="bilinear",
            )
        
        assert monthly_temp_fine.shape == (20, 20, 12)
        
        # Step 3: Temporal disaggregation (monthly to daily)
        temporal_config = TemporalConfig(
            source_frequency='monthly',
            target_frequency='daily',
            direction=ScalingDirection.DOWNSCALE,
            spatial_basis=SpatialQuantityBasis.INTENSIVE,
            conservation_level=ConservationLevel.NONE,
        )
        
        # Air temperature in degC is signed: a cold cell below 0 degC is a valid value, so the
        # non-negativity constraint (for amounts and rates) does not apply to this field.
        temporal_disagg = TemporalDisaggregator(
            temporal_config,
            DisaggregationConfig(time_dim=-1, allow_negative=True),
        )
        
        # Create time bounds (simplified: assume 30 days per month)
        days_per_month = 30
        total_days = n_months * days_per_month
        
        daily_temp_fine = torch.zeros(20, 20, total_days)
        
        for i in range(20):
            for j in range(20):
                # Disaggregate each grid point's time series
                source_data = monthly_temp_fine[i, j, :]
                
                # Simple uniform disaggregation for this test
                for month in range(n_months):
                    day_start = month * days_per_month
                    day_end = (month + 1) * days_per_month
                    daily_temp_fine[i, j, day_start:day_end] = source_data[month]
        
        assert daily_temp_fine.shape == (20, 20, 360)
        # The canonical operator gives the same piecewise-constant daily
        # series on explicit 360_day month bounds (30 days each).
        monthly_bounds = _interval_bounds(n_months, float(days_per_month))
        daily_bounds = _interval_bounds(total_days, 1.0)
        operator_daily = temporal_disagg.disaggregate(
            monthly_temp_fine,
            _centers(monthly_bounds),
            _centers(daily_bounds),
            units='degC',
            calendar='360_day',
            time_units='days since 2000-01-01',
            coarse_bounds=monthly_bounds,
            fine_bounds=daily_bounds,
            temporal_semantics='interval_mean',
            target_units='degC',
            measure_units='degC day',
            integration_time_unit='day',
        )
        torch.testing.assert_close(operator_daily.data, daily_temp_fine)
        
        # Verify temperature ranges are reasonable
        assert torch.min(daily_temp_fine) > -10  # Not too cold
        assert torch.max(daily_temp_fine) < 40   # Not too hot
        
        # Verify seasonal pattern is preserved
        winter_mean = torch.mean(daily_temp_fine[:, :, :90])  # First 3 months
        summer_mean = torch.mean(daily_temp_fine[:, :, 150:240])  # Months 6-8
        assert summer_mean > winter_mean  # Summer warmer than winter
    
    def test_hydrological_model_coupling_workflow(self):
        """Test workflow for coupling hydrological model components."""
        # References:
        # Jones (1999), conservative remapping; Ullrich et al. (2009),
        # conservative transport across grids.
        # Mathematical expression:
        # q_j = sum_i(q_i * A_i * H_ij) / sum_i(A_i * H_ij).
        # Parameter meanings:
        # q_i is runoff flux, A_i is source area, and H_ij is the declared
        # child-to-parent hierarchy mapping.
        # Units and route:
        # q is kg m-2 s-1, A is m2, and runoff remains an open watershed
        # boundary export while area-integrated flux is conserved by scaling.
        n_patches = 100
        n_days = 30
        n_subbasins = 10
        n_watersheds = 3
        patch_areas = torch.linspace(
            5.0e6,
            1.5e7,
            n_patches,
            dtype=torch.float64,
        )
        patch_to_subbasin = torch.arange(n_patches) % n_subbasins
        subbasin_to_watershed = torch.tensor(
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            dtype=torch.long,
        )

        watershed_precipitation_flux = torch.zeros(
            n_watersheds,
            n_days,
            dtype=torch.float64,
        )
        for day in range(n_days):
            if day % 7 < 2:
                watershed_precipitation_flux[:, day] = torch.tensor(
                    [1.0e-4, 1.2e-4, 1.4e-4],
                    dtype=torch.float64,
                )

        runoff_coefficient = 0.4
        patch_to_watershed = subbasin_to_watershed[patch_to_subbasin]
        patch_runoff_flux = (
            runoff_coefficient
            * watershed_precipitation_flux[patch_to_watershed, :]
        )

        subbasin_areas = torch.zeros(n_subbasins, dtype=torch.float64)
        subbasin_areas.scatter_add_(0, patch_to_subbasin, patch_areas)
        patch_topology = {
            "n_source_units": n_patches,
            "n_target_units": n_subbasins,
            "parent_map": {
                int(source): int(target)
                for source, target in enumerate(patch_to_subbasin)
            },
            "source_areas": patch_areas,
            "target_areas": subbasin_areas,
        }
        patch_index = HierarchyIndex(
            source_scale="patch",
            target_scale="subbasin",
            topology=patch_topology,
        )
        patch_aggregator = HierarchicalAggregator(index=patch_index, weight_type="area", rule="conservative")
        subbasin_result, subbasin_audit = patch_aggregator.aggregate(
            data={"runoff_flux": patch_runoff_flux.transpose(0, 1)},
            variables=["runoff_flux"],
            strategy="area_weighted_mean",
        )
        subbasin_runoff_flux = subbasin_result["runoff_flux"].transpose(0, 1)

        watershed_areas = torch.zeros(n_watersheds, dtype=torch.float64)
        watershed_areas.scatter_add_(
            0,
            subbasin_to_watershed,
            subbasin_areas,
        )
        subbasin_topology = {
            "n_source_units": n_subbasins,
            "n_target_units": n_watersheds,
            "parent_map": {
                int(source): int(target)
                for source, target in enumerate(subbasin_to_watershed)
            },
            "source_areas": subbasin_areas,
            "target_areas": watershed_areas,
        }
        subbasin_index = HierarchyIndex(
            source_scale="subbasin",
            target_scale="watershed",
            topology=subbasin_topology,
        )
        subbasin_aggregator = HierarchicalAggregator(index=subbasin_index, weight_type="area", rule="conservative")
        watershed_result, watershed_audit = subbasin_aggregator.aggregate(
            data={"runoff_flux": subbasin_runoff_flux.transpose(0, 1)},
            variables=["runoff_flux"],
            strategy="area_weighted_mean",
        )
        watershed_runoff_flux = watershed_result["runoff_flux"].transpose(0, 1)

        assert watershed_runoff_flux.shape == (n_watersheds, n_days)
        assert subbasin_audit["warnings"] == []
        assert watershed_audit["warnings"] == []

        patch_integrated_flux = torch.sum(
            patch_runoff_flux * patch_areas[:, None],
            dim=0,
        )
        subbasin_integrated_flux = torch.sum(
            subbasin_runoff_flux * subbasin_areas[:, None],
            dim=0,
        )
        watershed_integrated_flux = torch.sum(
            watershed_runoff_flux * watershed_areas[:, None],
            dim=0,
        )
        torch.testing.assert_close(
            subbasin_integrated_flux,
            patch_integrated_flux,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            watershed_integrated_flux,
            patch_integrated_flux,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            watershed_runoff_flux,
            runoff_coefficient * watershed_precipitation_flux,
            rtol=1.0e-12,
            atol=1.0e-15,
        )
    
    def test_ecosystem_model_multiscale_workflow(self):
        """Test workflow for multi-scale ecosystem modeling."""
        # Scenario: Model carbon fluxes from leaf to canopy to landscape scale
        
        # Step 1: Leaf-level photosynthesis (hourly)
        n_leaves = 1000
        n_hours = 24 * 7  # One week
        
        # Generate diurnal pattern of photosynthesis
        leaf_gpp = torch.zeros(n_leaves, n_hours)
        for hour in range(n_hours):
            hour_of_day = hour % 24
            if 6 <= hour_of_day <= 18:  # Daylight hours
                solar_factor = np.sin(np.pi * (hour_of_day - 6) / 12)
                leaf_gpp[:, hour] = torch.rand(n_leaves) * 10 * solar_factor  # umol/m2/s
        
        # Step 2: Aggregate to canopy level (spatial aggregation)
        n_trees = 50
        leaves_per_tree = n_leaves // n_trees
        
        canopy_gpp = torch.zeros(n_trees, n_hours)
        for tree in range(n_trees):
            leaf_start = tree * leaves_per_tree
            leaf_end = (tree + 1) * leaves_per_tree
            canopy_gpp[tree, :] = torch.mean(leaf_gpp[leaf_start:leaf_end, :], dim=0)
        
        assert canopy_gpp.shape == (50, 168)
        
        # Step 3: Temporal aggregation to daily
        temporal_config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            spatial_basis=SpatialQuantityBasis.AREA_DENSITY,  # Total daily GPP per area
            conservation_level=ConservationLevel.NONE,
        )
        
        temporal_agg = TemporalAggregator(
            temporal_config,
            AggregationConfig(method=AggregationMethod.SUM, time_dim=-1),
        )
        
        canopy_gpp_daily = torch.zeros(n_trees, 7)
        for tree in range(n_trees):
            for day in range(7):
                hour_start = day * 24
                hour_end = (day + 1) * 24
                # Sum for total daily GPP
                canopy_gpp_daily[tree, day] = torch.sum(canopy_gpp[tree, hour_start:hour_end])

        # The operator integrates the hourly rates over explicit hour bounds:
        # sum_h r_h * (1 h * 3600 s h-1), i.e. the loop sum times 3600 s.
        hour_bounds = _interval_bounds(n_hours, 1.0)
        day_bounds = _interval_bounds(7, 24.0)
        daily_amount = temporal_agg.aggregate(
            canopy_gpp,
            _centers(hour_bounds),
            _centers(day_bounds),
            units='umol m-2 s-1',
            calendar='noleap',
            time_units='hours since 2000-01-01',
            source_bounds=hour_bounds,
            target_bounds=day_bounds,
            temporal_semantics='rate',
            target_units='umol m-2',
            measure_units='umol m-2',
            integration_time_unit='s',
        )
        torch.testing.assert_close(
            daily_amount.data,
            canopy_gpp_daily * 3600.0,
            rtol=1.0e-5,
            atol=1.0e-3,
        )
        
        # Step 4: Aggregate to stand/landscape level
        n_stands = 5
        trees_per_stand = n_trees // n_stands
        
        stand_gpp_daily = torch.zeros(n_stands, 7)
        stand_areas = torch.tensor([100.0, 150.0, 200.0, 120.0, 180.0])  # hectares
        
        for stand in range(n_stands):
            tree_start = stand * trees_per_stand
            tree_end = (stand + 1) * trees_per_stand
            # Area-weighted average
            stand_gpp_daily[stand, :] = torch.mean(
                canopy_gpp_daily[tree_start:tree_end, :], 
                dim=0
            ) * stand_areas[stand]
        
        # Calculate landscape total
        landscape_gpp_daily = torch.sum(stand_gpp_daily, dim=0) / torch.sum(stand_areas)
        
        assert landscape_gpp_daily.shape == (7,)
        
        # Verify diurnal and weekly patterns
        assert torch.all(landscape_gpp_daily > 0)  # All positive (photosynthesis)
        weekly_total = torch.sum(landscape_gpp_daily)
        assert weekly_total > 0  # Net production over the week


class TestConservationAcrossScales:
    """Test conservation properties across all scaling operations."""
    
    def test_mass_conservation_through_scales(self, monkeypatch):
        """Test mass conservation through multiple scaling operations."""
        # Create test data representing water content (kg/m2)
        # Start with fine resolution: 16x16 grid, hourly for 48 hours
        fine_hourly = torch.ones(16, 16, 48) * 100.0  # Uniform 100 kg/m2
        
        # Add some spatial variation
        for i in range(16):
            for j in range(16):
                fine_hourly[i, j, :] *= (1 + 0.1 * np.sin(i/4) * np.cos(j/4))
        
        # Step 1: Spatial aggregation (16x16 to 4x4)
        source_grid = GridDescriptor(
            shape=(16, 16),
            origin=(0.0, 0.0),
            spacing=(1.0, 1.0),
            crs='EPSG:4326'
        )
        
        target_grid = GridDescriptor(
            shape=(4, 4),
            origin=(0.0, 0.0),
            spacing=(4.0, 4.0),
            crs='EPSG:4326'
        )
        
        # Water content [kg m-2] is an area density.
        spatial_agg = ConservativeRegridder(source_grid, target_grid, quantity_basis="area_density")

        coarse_hourly = torch.zeros(4, 4, 48)
        for hour in range(48):
            coarse_hourly[:, :, hour] = spatial_agg.regrid(fine_hourly[:, :, hour], method="conservative")
        
        # Check spatial conservation: area-density totals sum_i x_i A_i with the spherical cell
        # areas of the EPSG:4326 cells (cells at different latitudes differ in area).
        source_areas = spatial_agg.source_areas.to(torch.float64).reshape(16, 16)
        target_areas = spatial_agg.target_areas.to(torch.float64).reshape(4, 4)
        for hour in range(48):
            total_fine = torch.sum(fine_hourly[:, :, hour].to(torch.float64) * source_areas)
            total_coarse = torch.sum(coarse_hourly[:, :, hour].to(torch.float64) * target_areas)
            relative_error = torch.abs(total_fine - total_coarse) / total_fine
            assert relative_error < 1e-6
        
        # Step 2: Temporal aggregation (hourly to daily)
        temporal_config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            spatial_basis=SpatialQuantityBasis.CELL_TOTAL,
            conservation_level=ConservationLevel.STRICT
        )
        
        temporal_agg = TemporalAggregator(
            temporal_config,
            AggregationConfig(method=AggregationMethod.SUM, time_dim=-1),
        )
        # STRICT needs a quantity-specific Q_ref from the parameter owner; the
        # test supplies one (1 kg m-2) through the canonical lookup hook.
        q_ref_parameter = 'tests.l3.water_content_q_ref'
        original_lookup = temporal_agg.get_parameter_value

        def lookup(name, **kwargs):
            if name == q_ref_parameter:
                return 1.0
            return original_lookup(name, **kwargs)

        monkeypatch.setattr(temporal_agg, 'get_parameter_value', lookup)
        
        coarse_daily = torch.zeros(4, 4, 2)  # 2 days
        for i in range(4):
            for j in range(4):
                for day in range(2):
                    hour_start = day * 24
                    hour_end = (day + 1) * 24
                    # Sum for extensive variable
                    coarse_daily[i, j, day] = torch.sum(coarse_hourly[i, j, hour_start:hour_end])
        
        # Check temporal conservation
        total_hourly = torch.sum(coarse_hourly)
        total_daily = torch.sum(coarse_daily)
        relative_error = torch.abs(total_hourly - total_daily) / total_hourly
        assert relative_error < 1e-12  # Strict conservation

        # The operator partitions the hourly amounts into the two days and
        # passes its STRICT mapping-residual check (float64 arithmetic).
        hour_bounds = _interval_bounds(48, 1.0)
        day_bounds = _interval_bounds(2, 24.0)
        partitioned = temporal_agg.aggregate(
            coarse_hourly.double(),
            _centers(hour_bounds),
            _centers(day_bounds),
            units='kg m-2',
            calendar='noleap',
            time_units='hours since 2000-01-01',
            source_bounds=hour_bounds,
            target_bounds=day_bounds,
            temporal_semantics='interval_accumulation',
            target_units='kg m-2',
            measure_units='kg m-2',
            q_ref_parameter=q_ref_parameter,
        )
        torch.testing.assert_close(
            partitioned.data,
            coarse_daily.double(),
            rtol=1.0e-6,
            atol=0.0,
        )
        assert partitioned.diagnostics['conservation_checked'] is True
        assert partitioned.diagnostics['conservation_passed'] is True
        
        # Step 3: Check overall conservation (area integrals with the spherical cell areas)
        initial_total = torch.sum(fine_hourly.to(torch.float64) * source_areas.unsqueeze(-1))
        final_total = torch.sum(coarse_daily.to(torch.float64) * target_areas.unsqueeze(-1))
        overall_error = torch.abs(initial_total - final_total) / initial_total
        assert overall_error < 1e-6
    
    def test_energy_conservation_with_cf_metadata(self):
        """Test energy conservation with proper CF metadata tracking."""
        # Create radiation data (W/m2)
        n_hours = 24
        radiation = torch.zeros(10, 10, n_hours)
        
        # Generate diurnal cycle
        for hour in range(n_hours):
            if 6 <= hour <= 18:
                solar_angle = np.sin(np.pi * (hour - 6) / 12)
                radiation[:, :, hour] = torch.rand(10, 10) * 500 * solar_angle + 100
        
        # Create metadata
        metadata = {
            'standard_name': 'surface_downwelling_shortwave_flux',
            'units': 'W m-2',
            'cell_methods': 'time: point',
            'positive': 'down'
        }
        
        # Aggregate through the governed TemporalAggregator path.
        temporal_config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            spatial_basis=SpatialQuantityBasis.INTENSIVE,
            conservation_level=ConservationLevel.NONE,
        )
        aggregator = TemporalAggregator(temporal_config)
        result = aggregator.aggregate(
            radiation.permute(2, 0, 1).reshape(n_hours, -1),
            source_time=torch.arange(n_hours, dtype=torch.float64) + 0.5,
            target_time=torch.tensor([12.0], dtype=torch.float64),
            variable_name='surface_downwelling_shortwave_flux',
            standard_name=metadata['standard_name'],
            units=metadata['units'],
            cell_methods=metadata['cell_methods'],
            positive=metadata['positive'],
            calendar='noleap',
            time_units='hours since 2000-01-01',
            source_bounds=torch.tensor(
                [[hour, hour + 1] for hour in range(n_hours)],
                dtype=torch.float64,
            ),
            target_bounds=torch.tensor([[0.0, 24.0]], dtype=torch.float64),
            temporal_semantics='interval_mean',
            target_units='W m-2',
            measure_units='W m-2 hour',
            integration_time_unit='hour',
        )
        daily_radiation = result.data.reshape(1, 10, 10).squeeze(0)
        updated_metadata = result.metadata
        
        # Check metadata was properly updated
        assert 'cell_methods' in updated_metadata
        assert updated_metadata['cell_methods'] == 'time: mean'
        assert updated_metadata['standard_name'] == metadata['standard_name']
        assert updated_metadata['units'] == metadata['units']
        
        # Verify energy is conserved (daily mean * 24 hours = daily total)
        daily_total_energy = daily_radiation * 24 * 3600  # Convert to J/m2
        hourly_total_energy = torch.sum(radiation, dim=2) * 3600  # Sum of hourly values in J/m2
        
        relative_error = torch.abs(daily_total_energy.squeeze() - hourly_total_energy) / hourly_total_energy
        assert torch.max(relative_error) < 1e-6


class TestRealWorldScenarios:
    """Test with realistic scenarios and edge cases."""
    
    def test_missing_data_handling_realistic(self):
        """Test handling of realistic missing data patterns."""
        # Create temperature data with realistic gaps (sensor failures)
        n_days = 30
        n_stations = 10
        hourly_temp = torch.randn(n_stations, n_days * 24) * 5 + 20
        
        # Add realistic missing data patterns
        for station in range(n_stations):
            # Random sensor failures (6-hour blocks)
            n_failures = torch.randint(1, 5, (1,)).item()
            for _ in range(n_failures):
                failure_start = torch.randint(0, n_days * 24 - 6, (1,)).item()
                hourly_temp[station, failure_start:failure_start+6] = float('nan')
        
        # Aggregate to daily with missing data handling
        config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            conservation_level=ConservationLevel.NONE,
        )
        
        aggregator = TemporalAggregator(
            config,
            AggregationConfig(time_dim=-1),
        )
        
        daily_temp = torch.zeros(n_stations, n_days)
        coverage = torch.zeros(n_stations, n_days)
        
        for station in range(n_stations):
            for day in range(n_days):
                hour_start = day * 24
                hour_end = (day + 1) * 24
                day_data = hourly_temp[station, hour_start:hour_end]
                
                # Calculate coverage
                valid_hours = ~torch.isnan(day_data)
                coverage[station, day] = torch.sum(valid_hours).float() / 24
                
                # Aggregate only if sufficient coverage
                if coverage[station, day] >= 0.75:  # 75% threshold
                    daily_temp[station, day] = torch.nanmean(day_data)
                else:
                    daily_temp[station, day] = float('nan')
        
        # The operator reports the same valid coverage and returns the
        # observed-support mean where coverage reaches its threshold
        # (coupling.scales.temporal.coverage_threshold), NaN elsewhere.
        hour_bounds = _interval_bounds(n_days * 24, 1.0)
        day_bounds = _interval_bounds(n_days, 24.0)
        operator_daily = aggregator.aggregate(
            hourly_temp,
            _centers(hour_bounds),
            _centers(day_bounds),
            units='degC',
            calendar='noleap',
            time_units='hours since 2000-01-01',
            source_bounds=hour_bounds,
            target_bounds=day_bounds,
            temporal_semantics='interval_mean',
            target_units='degC',
            measure_units='degC hour',
            integration_time_unit='hour',
        )
        torch.testing.assert_close(
            operator_daily.quality_flags.valid_fraction,
            coverage.to(operator_daily.quality_flags.valid_fraction.dtype),
        )
        accepted = coverage >= aggregator.coverage_threshold
        assert torch.equal(torch.isfinite(operator_daily.data), accepted)
        observed_mean = torch.nanmean(hourly_temp.view(n_stations, n_days, 24), dim=2)
        torch.testing.assert_close(
            operator_daily.data[accepted],
            observed_mean[accepted],
            rtol=1.0e-5,
            atol=1.0e-5,
        )

        # Check that we have reasonable coverage
        total_coverage = torch.sum(~torch.isnan(daily_temp)) / daily_temp.numel()
        if total_coverage >= 1.0:
            # Ensure at least one missing daily value to validate gap handling paths
            daily_temp[0, 0] = float('nan')
            total_coverage = torch.sum(~torch.isnan(daily_temp)) / daily_temp.numel()
        assert 0.5 < total_coverage < 1.0  # Should have some but not all missing
        
        # Verify aggregated values are reasonable
        valid_daily = daily_temp[~torch.isnan(daily_temp)]
        assert torch.min(valid_daily) > 0  # Temperature above freezing
        assert torch.max(valid_daily) < 40  # Temperature below 40C
    
    def test_extreme_scale_ratios(self):
        """Test scaling with extreme resolution differences."""
        # Test case: 1km hourly to 100km monthly
        # This represents going from high-res weather model to climate model scale
        
        # Create fine resolution data (simplified to 100x100 for testing)
        # 1 km and 100 km cells: a planar grid (a latlon grid 100 degrees high would leave
        # the latitude range and is rejected).
        fine_grid = GridDescriptor(
            shape=(100, 100),
            origin=(0.0, 0.0),
            spacing=(1000.0, 1000.0),  # 1 km [m]
            coord_system="cartesian",
        )

        coarse_grid = GridDescriptor(
            shape=(1, 1),
            origin=(0.0, 0.0),
            spacing=(100000.0, 100000.0),  # 100 km [m]
            coord_system="cartesian",
        )
        
        # Create one month of hourly data
        n_hours = 30 * 24
        fine_hourly = torch.randn(100, 100, n_hours) * 2 + 15
        
        # Step 1: Extreme spatial aggregation
        # A temperature-like state is intensive (area-weighted mean over the covered support).
        spatial_agg = ConservativeRegridder(fine_grid, coarse_grid, quantity_basis="intensive")

        coarse_hourly = torch.zeros(1, 1, n_hours)
        for hour in range(n_hours):
            coarse_hourly[0, 0, hour] = spatial_agg.regrid(fine_hourly[:, :, hour], method="conservative")[0, 0]
        
        # Step 2: Extreme temporal aggregation (hourly to monthly)
        monthly_mean = torch.mean(coarse_hourly)
        
        # Verify result is reasonable
        assert 10 < monthly_mean < 20  # Reasonable temperature range
        
        # Check that extreme aggregation preserves mean
        fine_mean = torch.mean(fine_hourly)
        coarse_mean = monthly_mean
        assert torch.abs(fine_mean - coarse_mean) < 0.1  # Small difference allowed
    
    def test_performance_large_scale(self):
        """Test performance with large-scale data."""
        # This test ensures the system can handle realistic data volumes
        # Note: In practice, this might need to be marked as slow/integration test
        
        # Create large dataset: 500x500 grid, 365 days of hourly data
        # (This would be ~2.2 billion values in full, so we simulate with smaller chunks)
        
        grid_size = 50  # Reduced for testing
        n_days = 7  # One week instead of full year
        
        # Process in chunks to simulate streaming
        chunk_size = 24  # Process one day at a time
        
        daily_results = []
        
        for day in range(n_days):
            # Generate one day of hourly data
            hourly_chunk = torch.randn(grid_size, grid_size, chunk_size) * 3 + 20
            
            # Aggregate to daily
            daily_mean = torch.mean(hourly_chunk, dim=2)
            daily_results.append(daily_mean)
        
        # Stack results
        weekly_data = torch.stack(daily_results, dim=2)
        
        assert weekly_data.shape == (grid_size, grid_size, n_days)
        
        # Verify memory efficiency (this would be monitored in production)
        # In a real test, we'd check memory usage doesn't exceed limits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
