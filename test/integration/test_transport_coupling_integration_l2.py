"""
L2 Integration Tests for Transport and Coupling Modules

This module provides comprehensive integration tests that verify the interaction
between transport mechanisms and scaling/coupling components, ensuring proper
data flow, conservation across scales, and correct dependency handling.

Test Coverage:
- Transport process with temporal scaling integration
- Spatial scaling with transport mechanisms
- Multi-scale transport and coupling workflows
- Conservation across transport and scaling operations
- Parameter system integration across modules
- Dependency resolution between transport and coupling
- Error handling in integrated workflows
- Performance characteristics of integrated operations

Mathematical Verification:
- Transport equation with scaling: dc/dt + div(v * c) = S
- Conservation across scales: sum_fine(c_fine * V_fine) = sum_coarse(c_coarse * V_coarse)
- Temporal aggregation of transport fluxes: J_coarse = sum_j w_ij * J_fine
- Spatial scaling consistency: upscale(downscale(x)) ~= x

References:
- LeVeque, R.J. (1992). Numerical Methods for Conservation Laws
- Avissar, R. & Pielke, R.A. (1989). A parameterization of heterogeneous land surfaces
- CF Conventions Version 1.8: http://cfconventions.org/
"""

import os

# Import RELIC modules
import sys
from typing import Any, Dict

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.simulator import RELICSimulator
from relic.core.state import UpdateItem, UpdateMode
from relic.coupling.scales.spatial.spatial_scaling import SpatialScaling
from relic.coupling.scales.temporal import (
    ConservationLevel,
    ScalingDirection,
)
from relic.coupling.scales.types import SpatialQuantityBasis
from relic.coupling.scales.temporal.aggregator import (
    AggregationConfig,
    AggregationMethod,
    TemporalAggregator,
    TemporalConfig,
)
from relic.scientific.transport.advection import LinearAdvection


def _state_increment(updates: Dict[str, Any]) -> torch.Tensor:
    """Return the canonical per-step tracer increment from an UpdateItem."""
    item = updates["tracer_concentration"]
    assert isinstance(item, UpdateItem)
    assert item.mode is UpdateMode.ACCUMULATE
    assert item.per_timestep is True
    return item.value


def _temporal_config(**kwargs: Any) -> TemporalConfig:
    """Build the currently executable temporal configuration."""
    return TemporalConfig(
        conservation_level=ConservationLevel.NONE,
        **kwargs,
    )


def _aggregate_hourly_state(
    aggregator: TemporalAggregator,
    data: torch.Tensor,
    *,
    target_hours: int,
    variable_name: str = "concentration",
    standard_name: str = "tracer_concentration",
    units: str = "kg m-3",
    temporal_semantics: str = "interval_mean",
):
    """Aggregate one fixed hourly state window with explicit time contracts."""
    sample_count = int(data.shape[0])
    assert sample_count == target_hours
    return aggregator.aggregate(
        data=data,
        source_time=torch.arange(sample_count, dtype=torch.float32) + 0.5,
        target_time=torch.tensor([target_hours / 2.0], dtype=torch.float32),
        variable_name=variable_name,
        standard_name=standard_name,
        units=units,
        calendar="noleap",
        time_units="hours since 2000-01-01",
        source_bounds=torch.tensor(
            [[hour, hour + 1] for hour in range(sample_count)],
            dtype=torch.float32,
        ),
        target_bounds=torch.tensor(
            [[0.0, float(target_hours)]],
            dtype=torch.float32,
        ),
        temporal_semantics=temporal_semantics,
        target_units=units,
        measure_units=f"{units} hour",
        integration_time_unit="hour",
    )


class TestTransportTemporalScalingIntegration:
    """Test integration between transport processes and temporal scaling."""

    def test_advection_with_temporal_aggregation(self):
        """Test advection transport combined with temporal aggregation."""
        # Reference: LeVeque (1992), first-order upwind advection.
        # Expression: CFL = abs(u) * dt / dx = 0.5 for each hourly step.
        # Parameters: u is velocity [m s-1], dt is time [s], and dx is grid spacing [m].
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=3600.0,
            scheme="upwind"
        )

        # Set up velocity field
        velocity = torch.ones(10) * 0.5
        advection.set_velocity_field(velocity)

        # Create temporal aggregator
        temporal_config = _temporal_config(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE
        )
        # Concentration is an intensive state; aggregate with a mean.
        aggregator_conc = TemporalAggregator(
            temporal_config, AggregationConfig(method=AggregationMethod.TIME_WEIGHTED_MEAN)
        )
        # Simulate multiple time steps
        initial_concentration = torch.ones(10, dtype=torch.float32) * 2.0
        hourly_concentrations = []
        hourly_increments = []

        dt = 3600.0  # 1 hour
        for hour in range(24):
            state = {
                "tracer_concentration": initial_concentration.unsqueeze(0),
                "source_term": torch.zeros_like(initial_concentration.unsqueeze(0))
            }

            updates = advection.forward(state, time=hour * dt, dt=dt)

            # Update concentration
            delta_concentration = _state_increment(updates)
            initial_concentration = initial_concentration + delta_concentration.squeeze(0)

            # Store results
            hourly_concentrations.append(initial_concentration.clone())
            hourly_increments.append(delta_concentration.squeeze(0).clone())

        # Stack hourly results
        hourly_conc_tensor = torch.stack(hourly_concentrations)  # [24, 10]
        hourly_increment_tensor = torch.stack(hourly_increments)  # [24, 10]

        daily_conc_result = _aggregate_hourly_state(
            aggregator_conc,
            hourly_conc_tensor,
            target_hours=24,
        )

        daily_increment = torch.sum(hourly_increment_tensor, dim=0)

        # Verify results
        assert daily_conc_result.data.shape == (1, 10)  # 1 day, 10 spatial points
        assert torch.isfinite(daily_conc_result.data).all()
        assert daily_increment.shape == (10,)
        assert torch.isfinite(daily_increment).all()

        # Check conservation: daily mean should be reasonable
        expected_daily_mean = torch.mean(hourly_conc_tensor, dim=0)
        torch.testing.assert_close(daily_conc_result.data[0], expected_daily_mean, rtol=0.1, atol=0.0)

    def test_transport_with_different_temporal_methods(self):
        """Test transport with different temporal aggregation methods."""
        # Reference: LeVeque (1992), first-order upwind advection.
        # Expression: CFL = abs(u) * dt / dx = 0.3 for each hourly step.
        # Parameters: u is velocity [m s-1], dt is time [s], and dx is grid spacing [m].
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=3600.0,
            scheme="upwind"
        )

        velocity = torch.ones(5) * 0.3
        advection.set_velocity_field(velocity)

        # Test different aggregation methods
        methods = [
            AggregationMethod.MEAN,
            AggregationMethod.MAXIMUM,
            AggregationMethod.MINIMUM
        ]

        results = {}
        dt = 3600.0

        for method in methods:
            # Create aggregator with specific method
            temporal_config = _temporal_config(
                source_frequency='hourly',
                target_frequency='6hourly',
                direction=ScalingDirection.UPSCALE
            )
            agg_config = AggregationConfig(method=method)
            aggregator = TemporalAggregator(temporal_config, agg_config)

            # Run 6 hours of simulation
            # Use a non-uniform initial field so MEAN/MIN/MAX differ.
            concentration = torch.arange(1, 6, dtype=torch.float32)
            hourly_data = []

            for hour in range(6):
                state = {
                    "tracer_concentration": concentration.unsqueeze(0),
                    "source_term": torch.zeros_like(concentration.unsqueeze(0))
                }

                updates = advection.forward(state, time=hour * dt, dt=dt)
                delta_concentration = _state_increment(updates)
                concentration = concentration + delta_concentration.squeeze(0)
                hourly_data.append(concentration.clone())

            # Aggregate to 6-hourly
            hourly_tensor = torch.stack(hourly_data)
            result = _aggregate_hourly_state(
                aggregator,
                hourly_tensor,
                target_hours=6,
            )

            results[method] = result.data

            # Verify each result
            assert result.data.shape == (1, 5)
            assert torch.isfinite(result.data).all()

        # Different methods should produce different results
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                # Not all methods should be equal
                assert not torch.allclose(results[method1], results[method2])

        sum_aggregator = TemporalAggregator(
            temporal_config,
            AggregationConfig(method=AggregationMethod.SUM),
        )
        with pytest.raises(
            ValueError,
            match="sum is incompatible with interval_mean",
        ):
            _aggregate_hourly_state(
                sum_aggregator,
                hourly_tensor,
                target_hours=6,
            )

    def test_interval_total_temporal_scaling_is_fail_closed(self):
        """Transport conserves hourly mass while unbound SUM fails closed."""
        # Reference: LeVeque (1992), conservative first-order upwind advection.
        # Expression: CFL = abs(u) * dt / dx = 1.0 for each hourly step.
        # Parameters: u is velocity [m s-1], dt is time [s], and dx is grid spacing [m].
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=3600.0,
            scheme="upwind"
        )

        velocity = torch.ones(8) * 1.0
        advection.set_velocity_field(velocity)

        # Use extensive variable aggregation (sum)
        temporal_config = _temporal_config(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            spatial_basis=SpatialQuantityBasis.CELL_TOTAL  # kg per cell
        )
        agg_config = AggregationConfig(method=AggregationMethod.SUM)
        aggregator = TemporalAggregator(temporal_config, agg_config)

        # Simulate with conservation tracking
        initial_mass = torch.ones(8, dtype=torch.float32) * 10.0  # Mass in each cell
        dt = 3600.0
        hourly_masses = []

        for hour in range(24):
            state = {
                "tracer_concentration": initial_mass.unsqueeze(0),
                "source_term": torch.zeros_like(initial_mass.unsqueeze(0))
            }

            updates = advection.forward(state, time=hour * dt, dt=dt)
            delta_mass = _state_increment(updates)
            initial_mass = initial_mass + delta_mass.squeeze(0)
            hourly_masses.append(initial_mass.clone())

        # Check mass conservation in hourly data
        hourly_mass_tensor = torch.stack(hourly_masses)
        hourly_totals = torch.sum(hourly_mass_tensor, dim=1)  # Total mass each hour
        torch.testing.assert_close(
            hourly_totals,
            hourly_totals[0].expand_as(hourly_totals),
        )

        with pytest.raises(
            ValueError,
            match="interval_total temporal semantics are unsupported",
        ):
            _aggregate_hourly_state(
                aggregator,
                hourly_mass_tensor,
                target_hours=24,
                variable_name="mass",
                standard_name="mass_content",
                units="kg",
                temporal_semantics="interval_total",
            )


class TestTransportSpatialScalingIntegration:
    """Test integration between transport processes and spatial scaling."""

    def test_advection_with_spatial_upscaling(self):
        """Test advection transport combined with spatial upscaling."""
        # Create fine-resolution advection
        fine_advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(1.0, 1.0),
            scheme="upwind"
        )

        # Set up 2D velocity field
        u = torch.ones(8, 10) * 0.5  # x-velocity
        v = torch.ones(8, 10) * 0.3  # y-velocity
        fine_advection.set_velocity_field([u, v])

        # Create spatial upscaler
        spatial_upscaler = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0, 2.0),
            method="conservative",
            enforce_conservation=True
        )

        # Run fine-resolution transport
        fine_concentration = torch.ones(8, 10, dtype=torch.float32) * 2.0
        state = {
            "tracer_concentration": fine_concentration.unsqueeze(0),
            "source_term": torch.zeros_like(fine_concentration.unsqueeze(0))
        }

        # Reference: LeVeque (1992), multidimensional upwind stability.
        # Expression: CFL = (abs(u) / dx + abs(v) / dy) * dt = 0.8.
        # Parameters: u and v are velocities [m s-1], dx and dy are grid spacing [m],
        # and dt is the transport step [s].
        dt = 1.0
        updates = fine_advection.forward(state, time=0.0, dt=dt)
        fine_delta = _state_increment(updates)
        fine_new_conc = fine_concentration + fine_delta.squeeze(0)

        # Upscale to coarse resolution
        coarse_conc_before = spatial_upscaler.upscale(fine_concentration.unsqueeze(0))
        coarse_conc_after = spatial_upscaler.upscale(fine_new_conc.unsqueeze(0))

        # Compute coarse-level change
        coarse_delta = coarse_conc_after - coarse_conc_before

        # Verify conservation
        fine_total_change = torch.sum(fine_delta)
        coarse_total_change = torch.sum(coarse_delta)

        # Account for area ratio (4:1)
        expected_coarse_change = fine_total_change / 4.0
        torch.testing.assert_close(coarse_total_change, expected_coarse_change, rtol=0.01, atol=0.0)

    def test_advection_with_spatial_downscaling(self):
        """Test advection transport combined with spatial downscaling."""
        # Create coarse-resolution advection
        coarse_advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(2.0, 2.0),
            scheme="upwind"
        )

        # Set up coarse velocity field
        u = torch.ones(4, 5) * 0.5
        v = torch.ones(4, 5) * 0.3
        coarse_advection.set_velocity_field([u, v])

        # Create spatial downscaler
        spatial_downscaler = SpatialScaling(
            source_scale="coarse",
            target_scale="fine",
            source_resolution=(2.0, 2.0),
            target_resolution=(1.0, 1.0),
            method="conservative",
            enforce_conservation=True
        )
        spatial_upscaler = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0, 2.0),
            method="conservative",
            enforce_conservation=True
        )

        # Start with coarse field, downscale, run transport, upscale
        coarse_concentration = torch.ones(4, 5, dtype=torch.float32) * 3.0
        fine_concentration = spatial_downscaler.downscale(coarse_concentration.unsqueeze(0))

        # Run fine-resolution transport
        state = {
            "tracer_concentration": fine_concentration,
            "velocity_field": (
                u.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1),
                v.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1),
            ),
        }

        # Create fine-resolution advection for this step
        fine_advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(1.0, 1.0),
            scheme="upwind"
        )

        # Reference: LeVeque (1992), multidimensional upwind stability.
        # Expression: CFL = (abs(u) / dx + abs(v) / dy) * dt = 0.8.
        # Parameters: u and v are velocities [m s-1], dx and dy are grid spacing [m],
        # and dt is the transport step [s].
        dt = 1.0
        updates = fine_advection.forward(state, time=0.0, dt=dt)
        fine_delta = _state_increment(updates)
        fine_new_conc = fine_concentration + fine_delta

        # Upscale back to coarse resolution
        coarse_new_conc = spatial_upscaler.upscale(fine_new_conc)

        # Verify conservation through downscale-transport-upscale cycle
        original_coarse_total = torch.sum(coarse_concentration)
        final_coarse_total = torch.sum(coarse_new_conc.squeeze(0))

        # Should be approximately conserved
        relative_error = abs((final_coarse_total - original_coarse_total) / original_coarse_total)
        assert relative_error < 0.05  # Allow 5% error due to interpolation

    def test_multidimensional_transport_with_scaling(self):
        """Test 2D transport with the current spatial-scaling contract."""
        # Create 2D advection. SpatialScaling currently implements 2D
        # conservative remapping, so this verifies the active contract.
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(1.0, 1.0),
            scheme="upwind"
        )

        # Set up 2D velocity field.
        u = torch.ones(6, 6) * 0.4
        v = torch.ones(6, 6) * 0.3
        advection.set_velocity_field([u, v])

        # Create spatial upscaler for 2D.
        spatial_upscaler = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0, 2.0),
            method="conservative"
        )

        # Run transport
        fine_concentration = torch.ones(6, 6, dtype=torch.float32) * 1.5
        state = {
            "tracer_concentration": fine_concentration.unsqueeze(0),
            "source_term": torch.zeros_like(fine_concentration.unsqueeze(0))
        }

        # Reference: LeVeque (1992), multidimensional upwind stability.
        # Expression: CFL = (abs(u) / dx + abs(v) / dy) * dt = 0.7.
        # Parameters: u and v are velocities [m s-1], dx and dy are grid spacing [m],
        # and dt is the transport step [s].
        dt = 1.0
        updates = advection.forward(state, time=0.0, dt=dt)
        fine_new_conc = fine_concentration + _state_increment(updates).squeeze(0)

        # Upscale and verify
        coarse_conc = spatial_upscaler.upscale(fine_new_conc.unsqueeze(0))

        assert coarse_conc.shape == (1, 3, 3)  # 2D coarse grid
        assert torch.isfinite(coarse_conc).all()

        # Check area-weighted conservation.
        fine_total = torch.sum(fine_new_conc)
        coarse_total = torch.sum(coarse_conc)
        area_ratio = 4.0  # 2x2
        expected_coarse_total = fine_total / area_ratio
        torch.testing.assert_close(coarse_total, expected_coarse_total, rtol=0.01, atol=0.0)


class TestMultiScaleTransportWorkflow:
    """Test complete multi-scale transport and scaling workflows."""

    def test_spatiotemporal_transport_scaling(self):
        """Test combined spatial and temporal scaling with transport."""
        # Reference: LeVeque (1992), multidimensional upwind stability.
        # Expression: CFL = (abs(u) / dx + abs(v) / dy) * dt = 0.8.
        # Parameters: u and v are velocities [m s-1], dx and dy are grid spacing [m],
        # and dt is the hourly transport step [s].
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(3600.0, 3600.0),
            scheme="upwind"
        )

        u = torch.ones(6, 8) * 0.5
        v = torch.ones(6, 8) * 0.3
        advection.set_velocity_field([u, v])

        # Spatial upscaler
        spatial_upscaler = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(3600.0, 3600.0),
            target_resolution=(7200.0, 7200.0),
            method="conservative"
        )

        # Temporal aggregator
        temporal_config = _temporal_config(
            source_frequency='hourly',
            target_frequency='6hourly',
            direction=ScalingDirection.UPSCALE
        )
        agg_config = AggregationConfig(method=AggregationMethod.MEAN)
        temporal_aggregator = TemporalAggregator(temporal_config, agg_config)

        # Run multi-scale simulation
        fine_concentration = torch.ones(6, 8, dtype=torch.float32) * 2.0
        hourly_coarse_results = []

        for hour in range(6):
            # Fine-resolution transport
            state = {
                "tracer_concentration": fine_concentration.unsqueeze(0),
                "source_term": torch.zeros_like(fine_concentration.unsqueeze(0))
            }

            dt = 3600.0
            updates = advection.forward(state, time=hour * dt, dt=dt)
            fine_concentration = fine_concentration + _state_increment(updates).squeeze(0)

            # Spatial upscaling
            coarse_conc = spatial_upscaler.upscale(fine_concentration.unsqueeze(0))
            hourly_coarse_results.append(coarse_conc.squeeze(0))

        # Temporal aggregation of coarse results
        coarse_hourly_tensor = torch.stack(hourly_coarse_results)  # [6, 3, 4]
        six_hourly_result = _aggregate_hourly_state(
            temporal_aggregator,
            coarse_hourly_tensor,
            target_hours=6,
        )

        # Verify final result
        assert six_hourly_result.data.shape == (1, 3, 4)  # 1 time, 3x4 spatial
        assert torch.isfinite(six_hourly_result.data).all()

        # Check that the result is reasonable
        expected_mean = torch.mean(coarse_hourly_tensor, dim=0)
        torch.testing.assert_close(six_hourly_result.data[0], expected_mean, rtol=0.1, atol=0.0)

    def test_transport_scaling_dependency_chain(self):
        """Test transport processes in a dependency chain with scaling."""
        # Create multiple transport steps with different scales

        # Reference: LeVeque (1992), multidimensional upwind stability.
        # Expression: CFL = abs(u) * dt / dx = 0.8 for each hourly step.
        # Parameters: u is velocity [m s-1], dt is time [s], and dx is grid spacing [m].
        advection_fine = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(3600.0, 3600.0),
            scheme="upwind"
        )
        velocity_fine = [torch.ones(4, 20) * 0.8, torch.zeros(4, 20)]
        advection_fine.set_velocity_field(velocity_fine)

        # Step 2: Medium-resolution transport (after spatial scaling)
        spatial_upscaler = SpatialScaling(
            source_scale="fine",
            target_scale="medium",
            source_resolution=(3600.0, 3600.0),
            target_resolution=(7200.0, 7200.0),
            method="conservative"
        )

        # Step 3: Coarse-resolution transport (after temporal scaling)
        temporal_aggregator = TemporalAggregator(
            _temporal_config(
                source_frequency='hourly',
                target_frequency='daily',
                direction=ScalingDirection.UPSCALE
            ),
            AggregationConfig(method=AggregationMethod.MEAN)
        )

        # Run the chain
        fine_concentration = torch.linspace(1.0, 5.0, 80, dtype=torch.float32).reshape(4, 20)
        hourly_medium_results = []

        # Multiple hourly steps
        for hour in range(24):
            # Fine transport
            state = {
                "tracer_concentration": fine_concentration.unsqueeze(0),
                "source_term": torch.zeros_like(fine_concentration.unsqueeze(0))
            }

            dt = 3600.0
            updates = advection_fine.forward(state, time=hour * dt, dt=dt)
            fine_concentration = fine_concentration + _state_increment(updates).squeeze(0)

            # Spatial scaling to medium
            medium_conc = spatial_upscaler.upscale(fine_concentration.unsqueeze(0))
            hourly_medium_results.append(medium_conc.squeeze(0))

        # Temporal scaling to daily
        medium_tensor = torch.stack(hourly_medium_results)
        daily_result = _aggregate_hourly_state(
            temporal_aggregator,
            medium_tensor,
            target_hours=24,
        )

        # Verify chain integrity
        assert daily_result.data.shape == (1, 2, 10)  # 1 day, 2x10 medium grid
        assert torch.isfinite(daily_result.data).all()

        # Check conservation through the chain
        expected_daily_mean = torch.mean(medium_tensor, dim=0)
        medium_cell_area = spatial_upscaler.target_resolution[0] * spatial_upscaler.target_resolution[1]
        expected_total = torch.sum(expected_daily_mean) * medium_cell_area
        daily_total = torch.sum(daily_result.data[0]) * medium_cell_area
        relative_error = abs((daily_total - expected_total) / expected_total)
        assert relative_error < 0.1  # Allow 10% error

    def test_cross_scale_transport_consistency(self):
        """Test consistency of transport results across different scales."""
        # Create two advection instances at different resolutions
        fine_advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(0.5, 0.5),
            scheme="upwind"
        )

        coarse_advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(1.0, 1.0),
            scheme="upwind"
        )

        # Set up consistent velocity fields
        fine_velocity = [torch.ones(4, 20) * 0.05, torch.zeros(4, 20)]
        coarse_velocity = [torch.ones(2, 10) * 0.05, torch.zeros(2, 10)]
        fine_advection.set_velocity_field(fine_velocity)
        coarse_advection.set_velocity_field(coarse_velocity)

        # Initialize with consistent total mass
        fine_concentration = torch.ones(4, 20, dtype=torch.float32) * 2.0
        coarse_concentration = torch.ones(2, 10, dtype=torch.float32) * 2.0

        # Spatial scalers for comparison
        upscaler = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(0.5, 0.5),
            target_resolution=(1.0, 1.0),
            method="conservative"
        )

        downscaler = SpatialScaling(
            source_scale="coarse",
            target_scale="fine",
            source_resolution=(1.0, 1.0),
            target_resolution=(0.5, 0.5),
            method="conservative"
        )

        # Run simulation and compare results
        dt = 1.0

        # Fine-resolution simulation
        fine_state = {
            "tracer_concentration": fine_concentration.unsqueeze(0),
            "source_term": torch.zeros_like(fine_concentration.unsqueeze(0))
        }
        fine_updates = fine_advection.forward(fine_state, time=0.0, dt=dt)
        fine_new = fine_concentration + _state_increment(fine_updates).squeeze(0)

        # Coarse-resolution simulation
        coarse_state = {
            "tracer_concentration": coarse_concentration.unsqueeze(0),
            "source_term": torch.zeros_like(coarse_concentration.unsqueeze(0))
        }
        coarse_updates = coarse_advection.forward(coarse_state, time=0.0, dt=dt)
        coarse_new = coarse_concentration + _state_increment(coarse_updates).squeeze(0)

        # Cross-scale comparisons
        fine_upscaled = upscaler.upscale(fine_new.unsqueeze(0))
        coarse_downscaled = downscaler.downscale(coarse_new.unsqueeze(0))

        # Results should be approximately consistent
        torch.testing.assert_close(fine_upscaled, coarse_new.unsqueeze(0), rtol=0.1, atol=0.0)
        torch.testing.assert_close(coarse_downscaled, fine_new.unsqueeze(0), rtol=0.1, atol=0.0)


class TestParameterSystemIntegration:
    """Test parameter system integration across transport and coupling modules."""

    def test_shared_parameter_system(self):
        """Test shared parameter system between transport and coupling."""
        # Create components with shared parameters
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0
        )

        temporal_config = _temporal_config(
            source_frequency='hourly',
            target_frequency='daily',
        )
        aggregator = TemporalAggregator(temporal_config)

        # Verify both components use their canonical parameter interfaces.
        max_cfl = advection.get_parameter_value(
            "cfl_advection_max",
            units="dimensionless",
            namespace="transport.numerical",
        )
        assert 0.0 < float(max_cfl) <= 1.0
        assert aggregator.coverage_threshold == 0.8

    def test_parameter_validation_across_modules(self):
        """Test parameter validation across transport and coupling modules."""
        # Create transport with parameter validation
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0
        )

        # Create spatial scaling with same parameters
        spatial_scaling = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(1.0,),
            target_resolution=(2.0,),
            method="conservative",
        )

        # Verify parameter validation without restoring removed object caches.
        max_cfl = advection.get_parameter_value(
            "cfl_advection_max",
            units="dimensionless",
            namespace="transport.numerical",
        )
        assert 0.0 < float(max_cfl) <= 1.0
        assert spatial_scaling.source_resolution == (1.0,)


class TestSimulatorIntegration:
    """Test integration with RELIC simulator framework."""

    def test_transport_coupling_in_simulator(self):
        """Test transport and coupling components within simulator."""
        # Create simulator
        simulator = RELICSimulator(
            config={"enable_auto_scale": False, "strict_scale_detection": False}
        )

        # Reference: LeVeque (1992), first-order upwind advection.
        # Expression: CFL = abs(u) * dt / dx = 0.5 for the simulator step.
        # Parameters: u is velocity [m s-1], dt is time [s], and dx is grid spacing [m].
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1800.0,
            track_conservation=True,
        )

        # Create coupling component
        temporal_config = _temporal_config(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE
        )
        temporal_coupling = TemporalAggregator(temporal_config)
        temporal_coupling.name = "temporal_scaling"

        # Register the transport component and initialize simulator state. The
        # temporal aggregator requires explicit time-coordinate kwargs, so it is
        # exercised below on the simulator-produced transport update.
        simulator.register_functor(advection)

        # Create initial state
        initial_state = {
            "tracer_concentration": torch.ones(10, dtype=torch.float32) * 2.0,
            "source_term": torch.zeros(10, dtype=torch.float32),
            "velocity_field": torch.ones(10, dtype=torch.float32) * 0.5,
        }
        simulator.initialize_state(initial_state)

        # Run simulation step
        dt = 1800.0  # 30 minutes
        updates = simulator.step(dt=dt)

        # Verify simulator execution
        assert isinstance(updates, dict)
        assert "tracer_concentration" in updates

        state_increment = updates["tracer_concentration"]
        assert isinstance(state_increment, torch.Tensor)
        data = state_increment.unsqueeze(0).repeat(24, 1)
        temporal_coupling.to_device(data.device)
        aggregated = _aggregate_hourly_state(
            temporal_coupling,
            data,
            target_hours=24,
        )
        assert aggregated.data.device == data.device
        assert aggregated.data.shape == (1, 10)
        assert torch.isfinite(aggregated.data).all()

    def test_dependency_resolution_with_coupling(self):
        """Test dependency resolution including coupling components."""
        # Create dependency resolver
        resolver = DependencyGraphResolver()

        # Create transport with dependencies. The declared wind_model edge is
        # reported by get_dependencies, so a wind provider (stand-in advection
        # node) is registered for the required edge to resolve.
        wind_model = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            name="wind_model"
        )
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            name="advection"
        )
        advection.add_dependency("wind_model")  # Depends on wind

        # Create coupling component
        temporal_config = _temporal_config(
            source_frequency='sub_hourly',
            target_frequency='hourly',
            direction=ScalingDirection.UPSCALE
        )
        temporal_coupling = TemporalAggregator(temporal_config)
        temporal_coupling.name = "temporal_scaling"
        temporal_coupling.add_dependency("advection")  # Depends on advection

        # Add to resolver
        resolver.add_functor(advection, "advection")
        resolver.add_functor(temporal_coupling, "temporal_scaling")
        resolver.add_functor(wind_model, "wind_model")

        # Resolve dependencies
        execution_order = resolver.topological_sort()
        execution_groups = resolver.get_parallel_groups()

        # Verify dependency order
        assert "advection" in execution_order
        assert "temporal_scaling" in execution_order
        assert execution_order.index("wind_model") < execution_order.index("advection")

        # temporal_scaling should come after advection
        advection_idx = execution_order.index("advection")
        temporal_idx = execution_order.index("temporal_scaling")
        assert advection_idx < temporal_idx

        # The dependency also separates them into successive parallel levels.
        level_of = {
            node: level
            for level, group in enumerate(execution_groups)
            for node in group
        }
        assert level_of["advection"] < level_of["temporal_scaling"]


class TestErrorHandlingIntegration:
    """Test error handling in integrated transport-coupling workflows."""

    def test_transport_scaling_error_propagation(self):
        """Test error propagation between transport and scaling components."""
        # Create advection with problematic parameters
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0
        )

        # Set very high velocity to cause CFL violation
        problematic_velocity = torch.ones(5) * 100.0
        advection.set_velocity_field(problematic_velocity)

        # Create temporal aggregator
        temporal_config = _temporal_config(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE
        )
        aggregator = TemporalAggregator(temporal_config)

        # Run transport with error
        state = {
            "tracer_concentration": torch.ones(5, dtype=torch.float32),
            "source_term": torch.zeros(5, dtype=torch.float32)
        }

        # Reference: LeVeque (1992), explicit upwind CFL stability.
        # Expression: CFL = abs(u) * dt / dx; values above 1 fail fast.
        # Parameters: u is velocity [m s-1], dt is time [s], and dx is grid spacing [m].
        unsafe_dt = 1.0
        with pytest.raises(ValueError, match="CFL stability violation"):
            advection.forward(state, time=0.0, dt=unsafe_dt)

        stable_dt = 0.005
        updates = advection.forward(state, time=0.0, dt=stable_dt)
        assert "diagnostics.cfl_number" in updates
        assert torch.isfinite(updates["diagnostics.cfl_number"]).all()

        # Temporal aggregation should handle the results
        hourly_data = _state_increment(updates).unsqueeze(0).repeat(24, 1)
        result = _aggregate_hourly_state(
            aggregator,
            hourly_data,
            target_hours=24,
        )

        # Should produce valid results despite transport errors
        assert torch.isfinite(result.data).all()

    def test_dimension_mismatch_handling(self):
        """Test handling of dimension mismatches in integrated workflows."""
        # Create 2D advection
        advection_2d = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(1.0, 1.0),
            scheme="upwind"
        )

        # Create 1D spatial scaler (mismatch)
        spatial_scaler = SpatialScaling(
            source_scale="2d",
            target_scale="1d",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0,),
            method="conservative"
        )

        # Run 2D transport
        u = torch.ones(4, 5) * 0.5
        v = torch.ones(4, 5) * 0.3
        advection_2d.set_velocity_field([u, v])

        concentration_2d = torch.ones(4, 5, dtype=torch.float32) * 2.0
        state = {
            "tracer_concentration": concentration_2d.unsqueeze(0),
            "source_term": torch.zeros_like(concentration_2d.unsqueeze(0))
        }

        # Reference: LeVeque (1992), multidimensional upwind stability.
        # Expression: CFL = (abs(u) / dx + abs(v) / dy) * dt = 0.8.
        # Parameters: u and v are velocities [m s-1], dx and dy are grid spacing [m],
        # and dt is the transport step [s].
        dt = 1.0
        updates = advection_2d.forward(state, time=0.0, dt=dt)
        new_concentration_2d = concentration_2d + _state_increment(updates).squeeze(0)

        # Attempt spatial scaling with dimension mismatch
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            spatial_scaler.upscale(new_concentration_2d.unsqueeze(0))

    def test_conservation_violation_recovery(self):
        """Test recovery from conservation violations in integrated workflows."""
        # Create transport with conservation tracking
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            track_conservation=True
        )

        # Create spatial scaler with conservation enforcement
        spatial_scaler = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0, 1.0),
            method="conservative",
            enforce_conservation=True
        )

        # Run transport with artificial conservation violation
        velocity = torch.ones(10) * 0.5
        advection.set_velocity_field(velocity)

        concentration = torch.ones(10, dtype=torch.float32)
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        # Reference: LeVeque (1992), first-order upwind advection.
        # Expression: CFL = abs(u) * dt / dx = 0.5.
        # Parameters: u is velocity [m s-1], dt is time [s], and dx is grid spacing [m].
        dt = 1.0
        updates = advection.forward(state, time=0.0, dt=dt)

        # Check conservation diagnostics
        if "diagnostics.conservation_error" in updates:
            conservation_error = updates["diagnostics.conservation_error"]
            assert torch.isfinite(conservation_error)

        # Spatial scaling should enforce conservation
        new_concentration = concentration + _state_increment(updates).squeeze(0)
        coarse_concentration = spatial_scaler.upscale(new_concentration.unsqueeze(0))

        # Verify conservation enforcement
        fine_total = torch.sum(new_concentration)
        coarse_total = torch.sum(coarse_concentration)
        expected_coarse_total = fine_total / 2.0  # Area ratio
        torch.testing.assert_close(
            coarse_total,
            expected_coarse_total,
            rtol=0.01,
            atol=1e-6,
        )


class TestPerformanceIntegration:
    """Test performance characteristics of integrated workflows."""

    def test_memory_efficiency_spatiotemporal_scaling(self):
        """Test memory efficiency in spatiotemporal scaling workflows."""
        # Reference: LeVeque (1992), multidimensional upwind stability.
        # Expression: CFL = (abs(u) / dx + abs(v) / dy) * dt = 0.8.
        # Parameters: u and v are velocities [m s-1], dx and dy are grid spacing [m],
        # and dt is the one-hour transport step [s].
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(3600.0, 3600.0),
            scheme="upwind"
        )

        # Large velocity field
        u = torch.ones(50, 50) * 0.5
        v = torch.ones(50, 50) * 0.3
        advection.set_velocity_field([u, v])

        # Large concentration field
        concentration = torch.ones(50, 50, dtype=torch.float32) * 1.0

        # Create scalers
        spatial_scaler = SpatialScaling(
            source_scale="fine",
            target_scale="medium",
            source_resolution=(3600.0, 3600.0),
            target_resolution=(18000.0, 18000.0),
            method="conservative"
        )

        temporal_config = _temporal_config(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE
        )
        temporal_aggregator = TemporalAggregator(temporal_config)

        # Measure memory usage during workflow
        import gc

        # Baseline memory
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Run workflow
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 3600.0  # 1 hour
        updates = advection.forward(state, time=0.0, dt=dt)
        new_concentration = concentration + _state_increment(updates).squeeze(0)

        # Spatial scaling
        coarse_conc = spatial_scaler.upscale(new_concentration.unsqueeze(0))

        # Temporal scaling (simulate multiple steps)
        hourly_results = [coarse_conc.squeeze(0)]
        for _ in range(23):  # Total 24 hourly state snapshots
            state = {
                "tracer_concentration": new_concentration.unsqueeze(0),
                "source_term": torch.zeros_like(new_concentration.unsqueeze(0))
            }
            updates = advection.forward(state, time=0.0, dt=dt)
            new_concentration = new_concentration + _state_increment(updates).squeeze(0)
            coarse_conc = spatial_scaler.upscale(new_concentration.unsqueeze(0))
            hourly_results.append(coarse_conc.squeeze(0))

        # Temporal aggregation
        hourly_tensor = torch.stack(hourly_results)
        daily_result = _aggregate_hourly_state(
            temporal_aggregator,
            hourly_tensor,
            target_hours=24,
        )

        # Check memory efficiency
        gc.collect()
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        # Should not create excessive objects (indicative of memory leaks)
        assert object_increase < 1000  # Arbitrary threshold

        # Verify results
        assert daily_result.data.shape == (1, 10, 10)  # 1 day, medium resolution
        assert torch.isfinite(daily_result.data).all()

    def test_computational_efficiency_scaling_chain(self):
        """Test computational efficiency of scaling chains."""
        import time

        # Reference: LeVeque (1992), multidimensional upwind stability.
        # Expression: CFL = abs(u) * dt / dx = 0.5.
        # Parameters: u is velocity [m s-1], dt is the one-minute step [s],
        # and dx is grid spacing [m].
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(60.0, 60.0),
            scheme="upwind"
        )

        velocity = [torch.ones(10, 10) * 0.5, torch.zeros(10, 10)]
        advection.set_velocity_field(velocity)

        # Create scaling components
        spatial_scaler = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(60.0, 60.0),
            target_resolution=(300.0, 300.0),
            method="conservative"
        )

        temporal_config = _temporal_config(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE
        )
        temporal_aggregator = TemporalAggregator(temporal_config)

        # Timing test
        concentration = torch.ones(10, 10, dtype=torch.float32) * 2.0
        hourly_results = []

        start_time = time.time()

        # Run 24 transport samples for a daily temporal aggregation window.
        for minute in range(24):
            state = {
                "tracer_concentration": concentration.unsqueeze(0),
                "source_term": torch.zeros_like(concentration.unsqueeze(0))
            }

            dt = 60.0
            updates = advection.forward(state, time=minute * dt, dt=dt)
            concentration = concentration + _state_increment(updates).squeeze(0)

            coarse_conc = spatial_scaler.upscale(concentration.unsqueeze(0))
            hourly_results.append(coarse_conc.squeeze(0))

        # Temporal aggregation
        if hourly_results:
            hourly_tensor = torch.stack(hourly_results)
            daily_result = _aggregate_hourly_state(
                temporal_aggregator,
                hourly_tensor,
                target_hours=24,
            )

        end_time = time.time()
        computation_time = end_time - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert computation_time < 10.0  # 10 seconds for this test

        # Verify results
        if hourly_results:
            assert torch.isfinite(daily_result.data).all()


# Test fixtures
@pytest.fixture
def basic_transport_setup():
    """Fixture providing basic transport setup."""
    # Reference: LeVeque (1992), first-order upwind advection.
    # Expression: CFL = abs(u) * dt / dx = 0.5 for each hourly step.
    # Parameters: u is velocity [m s-1], dt is time [s], and dx is grid spacing [m].
    advection = LinearAdvection(
        spatial_dims=1,
        grid_spacing=3600.0,
        scheme="upwind"
    )
    velocity = torch.ones(10) * 0.5
    advection.set_velocity_field(velocity)

    return advection


@pytest.fixture
def basic_coupling_setup():
    """Fixture providing basic coupling setup."""
    temporal_config = _temporal_config(
        source_frequency='hourly',
        target_frequency='daily',
        direction=ScalingDirection.UPSCALE
    )
    return TemporalAggregator(temporal_config)


@pytest.fixture
def sample_concentration_data():
    """Fixture providing sample concentration data."""
    return torch.ones(10, dtype=torch.float32) * 2.0


# Integration tests using fixtures
def test_basic_transport_coupling_integration(
    basic_transport_setup,
    basic_coupling_setup,
    sample_concentration_data
):
    """Integration test for basic transport and coupling."""
    advection = basic_transport_setup
    aggregator = basic_coupling_setup
    concentration = sample_concentration_data.clone()

    # Run a full day of hourly transport steps so coverage meets the default threshold.
    hourly_concentrations = []
    dt = 3600.0
    for hour in range(24):
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0)),
        }
        updates = advection.forward(state, time=float(hour) * dt, dt=dt)
        concentration = concentration + _state_increment(updates).squeeze(0)
        hourly_concentrations.append(concentration.clone())

    hourly_data = torch.stack(hourly_concentrations)  # [24, 10]
    result = _aggregate_hourly_state(
        aggregator,
        hourly_data,
        target_hours=24,
    )

    # Verify integration
    assert torch.isfinite(result.data).all()
    assert result.data.shape == (1, 10)


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])

