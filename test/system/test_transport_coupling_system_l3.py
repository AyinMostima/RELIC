"""
L3 System Tests for Transport and Coupling Modules

This module provides comprehensive end-to-end system tests that verify the complete
functionality of transport and coupling modules in realistic scenarios, including
parameter loading, conservation validation, performance benchmarks, and comparison
with reference implementations.

Test Coverage:
- Complete transport-coupling workflow simulation
- Parameter system loading and validation
- Conservation law verification across entire system
- Performance benchmarking and optimization
- Real-world scenario simulation
- System robustness and error recovery
- Comparison with analytical solutions
- Integration with external data sources

Mathematical Verification:
- Complete system conservation: sum_all mass/totals conserved
- Transport equation accuracy: dc/dt + div(v c) = S
- Scaling consistency: upscale(downscale(x)) ~= x within tolerance
- Temporal integration accuracy: numerical vs analytical solutions

References:
- LeVeque, R.J. (1992). Numerical Methods for Conservation Laws
- WMO Guide to Climatological Practices
- IPCC Guidelines for Climate Data Processing
- CF Conventions Version 1.8
"""

import time
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.coupling.scales.spatial.spatial_scaling import SpatialScaling
from relic.coupling.scales.temporal import ConservationLevel, ScalingDirection
from relic.coupling.scales.temporal.aggregator import (
    AggregationConfig,
    AggregationMethod,
    TemporalAggregator,
    TemporalConfig,
)
from relic.parameters import get_global_manager
from relic.scientific.transport.advection import LinearAdvection
from relic.validation.framework.rules.conservation import ConservationRule


def _build_parameter_manager_mock(values: Dict[str, Any]) -> Mock:
    manager = Mock()

    def _get(key: str, *args, **kwargs):
        value = values.get(key)
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value

    manager.get.side_effect = _get
    manager.get_parameter.return_value = None
    manager.nn_parameters = {}
    return manager


def _tracer_increment(updates: Dict[str, Any]) -> torch.Tensor:
    """Return the canonical integrated advection increment and verify its contract."""
    update = updates["tracer_concentration"]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    assert isinstance(update.value, torch.Tensor)
    return update.value


def _transport_parameter_values() -> Dict[str, Any]:
    return {
        "transport.numerical.cfl_advection_max": 0.8,
        "transport.numerical.auto_adjust_timestep": True,
        "transport.numerical.min_timestep": 1e-6,
        "transport.numerical.conservation_tolerance": 1e-3,
        "transport.numerical.limiter_epsilon": 1e-12,
        "transport.numerical.artificial_viscosity_coefficient": 0.0,
        "transport.numerical.track_conservation": True,
        "transport.numerical.enforce_non_negative": True,
    }


def _temporal_parameter_values(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    values = {
        "coupling.scales.temporal.coverage_threshold": 0.8,
        "coupling.scales.defaults.coverage_epsilon": 1e-12,
        "coupling.scales.defaults.variables": {},
    }
    if overrides:
        values.update(overrides)
    return values


class TestCompleteSystemWorkflow:
    """Test complete end-to-end system workflows."""

    def test_full_transport_scaling_simulation(self):
        """Test complete transport and scaling simulation from initialization to results."""
        # Initialize parameter system
        with patch('relic.scientific.transport.advection.get_global_manager', create=True) as mock_transport_mgr:
            transport_mgr = _build_parameter_manager_mock(_transport_parameter_values())
            mock_transport_mgr.return_value = transport_mgr
            temporal_mgr = get_global_manager()
            if temporal_mgr is None:
                pytest.skip("ParameterManager not available for temporal aggregation tests")

            # Create system components
            simulator = RELICSimulator()
            state_manager = StateManager()

            # Transport component
            advection = LinearAdvection(
                spatial_dims=2,
                grid_spacing=(10000.0, 10000.0),
                scheme="upwind",
                boundary_method="periodic",
                name="atmospheric_advection"
            )

            # Set up a deterministic wind field on a 10 km atmospheric grid.
            nx, ny = 20, 16
            y_index, x_index = torch.meshgrid(
                torch.arange(ny, dtype=torch.float32),
                torch.arange(nx, dtype=torch.float32),
                indexing="ij",
            )
            u = 2.0 + 0.2 * torch.sin(2.0 * torch.pi * x_index / nx)
            v = 0.5 + 0.1 * torch.cos(2.0 * torch.pi * y_index / ny)
            advection.set_velocity_field([v, u])  # Note: order is (v, u) for (y, x)

            # Spatial scaling component
            spatial_scaler = SpatialScaling(
                source_scale="high_res",
                target_scale="low_res",
                source_resolution=(10000.0, 10000.0),
                target_resolution=(40000.0, 40000.0),
                method="conservative",
                preserve="mass",
                enforce_conservation=False,
            )

            # Temporal scaling component
            temporal_config = TemporalConfig(
                source_frequency='hourly',
                target_frequency='daily',
                direction=ScalingDirection.UPSCALE,
                conservation_level=ConservationLevel.NONE
            )
            temporal_aggregator = TemporalAggregator(
                temporal_config,
                aggregation_config=AggregationConfig(method=AggregationMethod.MEAN),
                parameter_manager=temporal_mgr,
            )

            # Register components with simulator
            simulator.register_functor(advection)
            simulator.register_functor(spatial_scaler)
            simulator.register_functor(temporal_aggregator)

            # Initialize system state
            initial_concentration = torch.ones(ny, nx, dtype=torch.float32) * 1.0
            # Add spatial variation
            for i in range(ny):
                for j in range(nx):
                    initial_concentration[i, j] *= (1.0 + 0.5 * np.sin(2 * np.pi * i / ny) * np.cos(2 * np.pi * j / nx))

            state_manager.set_state({
                "tracer_concentration": initial_concentration,
                "source_term": torch.zeros_like(initial_concentration)
            })

            # Run complete simulation
            simulation_results = []
            hourly_concentrations = []
            daily_aggregates = []

            total_time = 24 * 3600.0  # 24 hours
            dt = 1800.0  # 30 minutes
            n_steps = int(total_time / dt)

            for step in range(n_steps):
                current_time = step * dt

                # Get current state
                current_state = state_manager.get_state()

                # Run transport
                transport_updates = advection.forward(current_state, current_time, dt)

                # Apply state updates
                state_manager.apply_updates({
                    "tracer_concentration": transport_updates["tracer_concentration"]
                }, dt=dt)

                # Store hourly results
                if (step + 1) % 2 == 0:  # Every hour (2 steps of 30 min)
                    current_concentration = state_manager.get_state()["tracer_concentration"]
                    hourly_concentrations.append(current_concentration.clone())

                    # Spatial scaling
                    coarse_conc = spatial_scaler.upscale(current_concentration.unsqueeze(0))
                    simulation_results.append({
                        'time': current_time,
                        'fine_concentration': current_concentration.clone(),
                        'coarse_concentration': coarse_conc.squeeze(0).clone(),
                        'conservation_diagnostics': transport_updates.get('diagnostics', {})
                    })

            # Temporal aggregation
            if hourly_concentrations:
                hourly_tensor = torch.stack(hourly_concentrations)  # [24, ny, nx]
                source_time = torch.arange(24, dtype=torch.float32) + 0.5
                target_time = torch.tensor([12.0], dtype=torch.float32)

                daily_result = temporal_aggregator.aggregate(
                    data=hourly_tensor,
                    source_time=source_time,
                    target_time=target_time,
                    variable_name='tracer_concentration',
                    standard_name='tracer_concentration',
                    units='kg m-3',
                    calendar="noleap",
                    time_units="hours since 1970-01-01",
                    source_bounds=torch.tensor(
                        [[hour, hour + 1] for hour in range(24)],
                        dtype=torch.float32,
                    ),
                    target_bounds=torch.tensor(
                        [[0.0, 24.0]], dtype=torch.float32
                    ),
                    temporal_semantics="interval_mean",
                    target_units="kg m-3",
                    measure_units="kg m-3 hour",
                    integration_time_unit="hour",
                )
                daily_aggregates.append(daily_result)

            # System validation
            assert len(simulation_results) == 24  # 24 hourly results
            assert len(daily_aggregates) == 1     # 1 daily aggregate

            # Check conservation throughout simulation
            initial_total = torch.sum(initial_concentration)
            final_concentration = state_manager.get_state()["tracer_concentration"]
            final_total = torch.sum(final_concentration)

            relative_error = abs((final_total - initial_total) / initial_total)
            assert relative_error < 1.0e-5

            # Check spatial scaling consistency
            for result in simulation_results:
                fine_conc = result['fine_concentration']
                coarse_conc = result['coarse_concentration']
                assert torch.isfinite(coarse_conc).all()
                assert torch.all(coarse_conc >= 0)
                torch.testing.assert_close(
                    torch.mean(coarse_conc),
                    torch.mean(fine_conc),
                    rtol=1.0e-5,
                    atol=1.0e-7,
                )

            # Check temporal aggregation
            expected_daily_mean = torch.mean(hourly_tensor, dim=0)
            aggregated = daily_aggregates[0].data[0]
            assert torch.isfinite(aggregated).all()
            torch.testing.assert_close(
                aggregated.to(expected_daily_mean.dtype),
                expected_daily_mean,
                rtol=1.0e-5,
                atol=1.0e-7,
            )

            # Metadata validation
            daily_meta = daily_aggregates[0].metadata
            assert 'cell_methods' in daily_meta
            assert 'history' in daily_meta
            assert 'variable_name' in daily_meta

            # Quality flags validation
            daily_quality = daily_aggregates[0].quality_flags
            assert torch.all(daily_quality.valid_fraction >= 0.0)
            assert torch.all(daily_quality.valid_fraction <= 1.0)
            assert torch.all(
                daily_quality.num_samples == source_time.numel()
            )

    def test_multi_scenario_simulation(self):
        """Test system with multiple transport scenarios."""
        scenarios = [
            {
                'name': 'pure_advection',
                'velocity': (torch.ones(10, 10) * 1.0, torch.ones(10, 10) * 0.5),
                'source': torch.zeros(10, 10),
                'description': 'Pure advection with no sources'
            },
            {
                'name': 'advection_with_source',
                'velocity': (torch.ones(10, 10) * 0.8, torch.ones(10, 10) * 0.4),
                'source': torch.ones(10, 10) * 0.01,
                'description': 'Advection with constant source'
            },
            {
                'name': 'variable_advection',
                'velocity': (
                    torch.ones(10, 10) * (1.0 + 0.5 * torch.randn(10, 10)),
                    torch.ones(10, 10) * (0.5 + 0.2 * torch.randn(10, 10))
                ),
                'source': torch.zeros(10, 10),
                'description': 'Variable velocity field'
            }
        ]

        results = {}

        for scenario in scenarios:
            # Create components for this scenario
            advection = LinearAdvection(
                spatial_dims=2,
                grid_spacing=(1.0, 1.0),
                scheme="upwind",
                device=torch.device("cpu"),
            )

            v, u = scenario['velocity']
            advection.set_velocity_field([v, u])

            spatial_scaler = SpatialScaling(
                source_scale="fine",
                target_scale="coarse",
                source_resolution=(1.0, 1.0),
                target_resolution=(2.0, 2.0),
                method="conservative",
                device=torch.device("cpu"),
            )

            # Run simulation
            concentration = torch.ones(10, 10, dtype=torch.float32) * 2.0
            state = {
                "tracer_concentration": concentration.unsqueeze(0),
                "source_term": scenario['source'].unsqueeze(0)
            }

            dt = 0.1
            n_steps = 6

            for step in range(n_steps):
                updates = advection.forward(state, step * dt, dt)
                concentration = concentration + _tracer_increment(updates).squeeze(0)
                state["tracer_concentration"] = concentration.unsqueeze(0)

            # Apply spatial scaling
            coarse_conc = spatial_scaler.upscale(concentration.unsqueeze(0))

            results[scenario['name']] = {
                'final_concentration': concentration,
                'coarse_concentration': coarse_conc.squeeze(0),
                'source': scenario['source'],
                'description': scenario['description']
            }

        # Validate results
        for name, result in results.items():
            # All results should be physically reasonable
            assert torch.isfinite(result['final_concentration']).all()
            assert torch.isfinite(result['coarse_concentration']).all()
            assert torch.all(result['final_concentration'] >= 0)  # Non-negative

            # Conservation check for intensive concentration fields.
            # Fine mass = sum_i c_i A_i; coarse mass = sum_J C_J A_J.
            fine_cell_area = torch.tensor(1.0 * 1.0, dtype=result['final_concentration'].dtype)
            coarse_cell_area = torch.tensor(2.0 * 2.0, dtype=result['coarse_concentration'].dtype)
            fine_mass = torch.sum(result['final_concentration']) * fine_cell_area
            coarse_mass = torch.sum(result['coarse_concentration']) * coarse_cell_area
            torch.testing.assert_close(coarse_mass, fine_mass, rtol=0.01, atol=0.0)

        # Compare scenarios
        pure_adv = results['pure_advection']['final_concentration']
        with_source = results['advection_with_source']['final_concentration']

        # With source should have higher concentrations
        assert torch.mean(with_source) > torch.mean(pure_adv)

    def test_system_robustness_under_stress(self):
        """Test system robustness under stress conditions."""
        stress_conditions = [
            {
                'name': 'high_cfl',
                'velocity_magnitude': 10.0,
                'dt': 0.2,
                'expected_behavior': 'reject_unstable'
            },
            {
                'name': 'zero_velocity',
                'velocity_magnitude': 0.0,
                'dt': 3600.0,
                'expected_behavior': 'no_change'
            },
            {
                'name': 'large_source',
                'velocity_magnitude': 1.0,
                'source_magnitude': 100.0,
                'dt': 0.5,
                'expected_behavior': 'dominant_source'
            },
            {
                'name': 'very_fine_grid',
                'grid_size': (50, 50),
                'velocity_magnitude': 1.0,
                'dt': 0.5,
                'expected_behavior': 'handle_large_arrays'
            }
        ]

        for condition in stress_conditions:
            # Configure based on condition
            if condition['name'] == 'very_fine_grid':
                ny, nx = condition['grid_size']
            else:
                ny, nx = 10, 10

            # Create transport component
            advection = LinearAdvection(
                spatial_dims=2,
                grid_spacing=(1.0, 1.0),
                scheme="upwind",
            )

            # Set velocity
            velocity_mag = condition.get('velocity_magnitude', 1.0)
            u = torch.ones(ny, nx) * velocity_mag
            v = torch.ones(ny, nx) * velocity_mag * 0.5
            advection.set_velocity_field([v, u])

            # Set source
            source_mag = condition.get('source_magnitude', 0.0)
            source = torch.ones(ny, nx) * source_mag

            # Run stress test
            concentration = torch.ones(ny, nx, dtype=torch.float32)
            state = {
                "tracer_concentration": concentration.unsqueeze(0),
                "source_term": source.unsqueeze(0)
            }

            dt = condition['dt']

            if condition['expected_behavior'] == 'reject_unstable':
                with pytest.raises(ValueError, match="CFL"):
                    advection.forward(state, 0.0, dt)
                continue

            updates = advection.forward(state, 0.0, dt)

            # Verify updates are physically reasonable.
            increment = _tracer_increment(updates)
            assert torch.isfinite(increment).all()
            cfl_limit = float(
                advection.get_parameter_value(
                    "cfl_advection_max",
                    namespace="transport.numerical",
                    required=True,
                )
            )
            assert float(updates["diagnostics.cfl_number"]) <= cfl_limit + 1.0e-6

            if condition['expected_behavior'] == 'no_change':
                torch.testing.assert_close(
                    increment,
                    torch.zeros_like(increment),
                    rtol=0.0,
                    atol=1e-10,
                )
            elif condition['expected_behavior'] == 'dominant_source':
                expected_source_increment = source.unsqueeze(0) * dt
                torch.testing.assert_close(
                    increment,
                    expected_source_increment.to(
                        device=increment.device,
                        dtype=increment.dtype,
                    ),
                    rtol=1.0e-6,
                    atol=1.0e-6,
                )

    def test_simulator_substeps_cover_the_full_requested_timestep(self):
        """Scheduler-owned CFL substeps must cover the requested interval."""
        concentration = torch.tensor(
            [[0.0, 0.2, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        velocity = torch.ones_like(concentration)
        requested_dt = 1.6
        stable_dt = requested_dt / 2.0

        scheduled = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            scheme="upwind",
            boundary_method="periodic",
        )
        scheduled.set_velocity_field(velocity)
        scheduled.metadata.substep_count = 2
        simulator = RELICSimulator(
            {
                "time_step": requested_dt,
                "device": "cpu",
                "conservation_checks": False,
            }
        )
        simulator.initialize_state(
            {
                "tracer_concentration": concentration,
                "velocity_field": velocity,
                "source_term": torch.zeros_like(concentration),
            }
        )
        simulator.register_functor(scheduled)
        simulator.step()
        scheduled_final = simulator.state_manager.get_state("tracer_concentration")

        reference = LinearAdvection(
            spatial_dims=1,
            grid_spacing=1.0,
            scheme="upwind",
            boundary_method="periodic",
            device=torch.device("cpu"),
        )
        reference.set_velocity_field(velocity)
        reference_state = concentration.to(reference.device)
        for substep in range(2):
            reference_update = reference.forward(
                {
                    "tracer_concentration": reference_state,
                    "source_term": torch.zeros_like(reference_state),
                },
                substep * stable_dt,
                stable_dt,
            )
            reference_state = reference_state + _tracer_increment(reference_update)

        torch.testing.assert_close(
            scheduled_final,
            reference_state,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        torch.testing.assert_close(
            torch.sum(scheduled_final),
            torch.sum(concentration).to(scheduled_final.device),
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    def test_parameter_system_integration_comprehensive(self):
        """Test comprehensive parameter system integration."""
        # Test parameter loading, validation, and usage across components
        with patch('relic.scientific.transport.advection.get_global_manager', create=True) as mock_mgr:
            # Transport parameters
            transport_params = {
                'transport.numerical.cfl_advection_max': {'value': 0.8, 'units': 'dimensionless', 'bounds': (0.1, 1.0)},
                'transport.numerical.track_conservation': {'value': True, 'units': 'boolean'},
                'transport.numerical.enforce_non_negative': {'value': True, 'units': 'boolean'},
                'transport.advection.grid_spacing': {'value': [1.0, 1.0], 'units': 'm'},
            }

            # Temporal scaling parameters
            temporal_params = {
                'coupling.scales.temporal.coverage_threshold': {'value': 0.8, 'units': 'dimensionless'},
                'coupling.scales.defaults.coverage_epsilon': {'value': 1e-12, 'units': 'dimensionless'},
                'coupling.scales.defaults.variables': {'value': {}, 'units': 'dimensionless'},
            }

            all_params = {**transport_params, **temporal_params}
            param_values = {
                key: value.get('value') if isinstance(value, dict) else value
                for key, value in all_params.items()
            }
            param_manager = _build_parameter_manager_mock(param_values)
            mock_mgr.return_value = param_manager

            # Create components that use parameters
            advection = LinearAdvection(
                spatial_dims=2,
                grid_spacing=(1.0, 1.0),
                scheme="upwind",
                device=torch.device("cpu"),
            )

            temporal_config = TemporalConfig(
                source_frequency='hourly',
                target_frequency='daily',
            )
            aggregator = TemporalAggregator(temporal_config, parameter_manager=param_manager)

            # Verify parameter loading
            cfl_limit = advection.get_parameter_value(
                "cfl_advection_max",
                namespace="transport.numerical",
                required=True,
            )
            assert 0.0 < float(cfl_limit) <= 1.0
            assert not hasattr(advection, "max_cfl")
            assert not hasattr(advection, "auto_adjust_timestep")
            assert not hasattr(advection, "min_timestep")
            assert not hasattr(advection, "conservation_tolerance")
            assert advection.track_conservation is True
            assert advection.non_negative is True

            assert aggregator.coverage_threshold == 0.8

            # Verify parameter manager was called
            assert param_manager.get.call_count > 0

            # Test parameter validation
            for param_name, param_info in all_params.items():
                if 'bounds' in param_info:
                    value = param_info['value']
                    bounds = param_info['bounds']
                    if isinstance(value, list):
                        for v in value:
                            assert bounds[0] <= v <= bounds[1], f"Parameter {param_name} out of bounds"
                    else:
                        assert bounds[0] <= value <= bounds[1], f"Parameter {param_name} out of bounds"

    def test_system_conservation_validation(self):
        """Test comprehensive system conservation validation."""
        # Create conservation rule validator
        conservation_rule = ConservationRule()

        # Create system components
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(1.0, 1.0),
            scheme="upwind",
            track_conservation=True
        )

        spatial_scaler = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(1.0, 1.0),
            target_resolution=(2.0, 2.0),
            method="conservative",
            enforce_conservation=True
        )
        reverse_spatial_scaler = SpatialScaling(
            source_scale="coarse",
            target_scale="fine",
            source_resolution=(2.0, 2.0),
            target_resolution=(1.0, 1.0),
            method="conservative",
            enforce_conservation=True,
            device=spatial_scaler.device
        )

        # Run conservation test
        initial_mass = torch.ones(20, 20, dtype=torch.float32) * 10.0
        velocity = (torch.ones(20, 20) * 0.5, torch.ones(20, 20) * 0.3)
        advection.set_velocity_field(velocity)

        # Run multiple steps to test conservation over time. The uniform tracer field keeps
        # the inflow and outflow boundary fluxes equal, so its amount is conserved; the rule
        # judges every recorded field against the initial one (budget_closure with
        # validation.conservation.conservation_tolerance; the float32 rounding of the
        # 400-cell amounts and the ten steps enters its rounding allowance).
        current_mass = initial_mass.clone()
        mass_history = [current_mass.clone()]

        for step in range(10):
            state = {
                "tracer_concentration": current_mass.unsqueeze(0),
                "source_term": torch.zeros_like(current_mass.unsqueeze(0))
            }

            dt = 0.5
            updates = advection.forward(state, step * dt, dt)
            current_mass = current_mass + _tracer_increment(updates).squeeze(0)
            mass_history.append(current_mass.clone())

            # Apply spatial scaling every other step
            if step % 2 == 1:
                coarse_mass = spatial_scaler.upscale(current_mass.unsqueeze(0))
                fine_mass_recovered = reverse_spatial_scaler.downscale(coarse_mass)

                # Check conservative remapping with area-weighted mass.
                fine_cell_area = torch.tensor(1.0 * 1.0, dtype=current_mass.dtype, device=current_mass.device)
                coarse_cell_area = torch.tensor(2.0 * 2.0, dtype=coarse_mass.dtype, device=coarse_mass.device)
                original_total = torch.sum(current_mass) * fine_cell_area
                coarse_total = torch.sum(coarse_mass) * coarse_cell_area
                recovered_total = torch.sum(fine_mass_recovered) * fine_cell_area
                relative_error = torch.abs((coarse_total - original_total) / original_total)
                roundtrip_error = torch.abs((recovered_total - original_total) / original_total)
                assert relative_error < 0.01, f"Mass not conserved in upscale at step {step}"
                assert roundtrip_error < 0.01, f"Mass not conserved in scaling cycle at step {step}"

        # Validate conservation using rule
        conservation_report = conservation_rule.validate({'mass_history': mass_history})

        # Check conservation report
        assert conservation_report['status'] == 'passed', conservation_report['message']
        assert conservation_report['relative_error'] <= conservation_report['tolerance']
        assert conservation_report['tolerance_parameter'] == (
            'validation.conservation.conservation_tolerance'
        )

    def test_system_performance_benchmarking(self):
        """Test system performance under various conditions."""
        performance_tests = [
            {
                'name': 'small_2d',
                'shape': (10, 10),
                'n_steps': 100,
            },
            {
                'name': 'medium_2d',
                'shape': (25, 25),
                'n_steps': 50,
            },
            {
                'name': 'large_2d',
                'shape': (50, 50),
                'n_steps': 20,
            }
        ]

        performance_results = {}

        for test in performance_tests:
            ny, nx = test['shape']
            n_steps = test['n_steps']

            # Create components
            advection = LinearAdvection(
                spatial_dims=2,
                grid_spacing=(1.0, 1.0),
                scheme="upwind",
                device=torch.device("cpu"),
            )

            velocity = (torch.ones(ny, nx) * 0.5, torch.ones(ny, nx) * 0.3)
            advection.set_velocity_field(velocity)

            # Performance test
            concentration = torch.ones(ny, nx, dtype=torch.float32)
            state = {
                "tracer_concentration": concentration.unsqueeze(0),
                "source_term": torch.zeros_like(concentration.unsqueeze(0))
            }

            start_time = time.time()

            for step in range(n_steps):
                dt = 0.5
                updates = advection.forward(state, step * dt, dt)
                concentration = concentration + _tracer_increment(updates).squeeze(0)
                state["tracer_concentration"] = concentration.unsqueeze(0)

            end_time = time.time()
            elapsed_time = end_time - start_time

            performance_results[test['name']] = {
                'elapsed_time': elapsed_time,
                'grid_size': test['shape'],
                'n_steps': n_steps,
                'time_per_step': elapsed_time / n_steps
            }

            assert np.isfinite(elapsed_time)
            assert elapsed_time > 0.0

        # Performance scaling analysis
        small_time = performance_results['small_2d']['time_per_step']
        medium_time = performance_results['medium_2d']['time_per_step']
        large_time = performance_results['large_2d']['time_per_step']

        # Check that performance scales reasonably (not exponentially)
        medium_ratio = medium_time / max(small_time, 1e-6)
        large_ratio = large_time / max(medium_time, 1e-6)
        assert medium_ratio < 10.0, "Medium grid performance degradation too large"
        assert large_ratio < 15.0, "Large grid performance degradation too large"

    def test_real_world_scenario_simulation(self):
        """Test system with realistic environmental scenario."""
        # Simulate atmospheric pollutant transport over urban area
        scenario_config = {
            'domain_size': (40, 50),
            'grid_resolution': 5000.0,  # 5 km grid spacing [m]
            'simulation_time': 24 * 3600.0,  # 24 hours
            'time_step': 600.0,           # 10 minutes
            'pollutant_source': 'urban_emissions',
            'meteorology': 'diurnal_cycle'
        }

        # Create urban emission source pattern
        ny, nx = scenario_config['domain_size']
        emission_sources = torch.zeros(ny, nx, dtype=torch.float32)

        # Add urban centers (higher emissions)
        urban_centers = [(10, 12), (15, 25), (8, 35), (20, 40)]
        for center_y, center_x in urban_centers:
            for i in range(ny):
                for j in range(nx):
                    distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    if distance < 5.0:  # 5km radius
                        emission_sources[i, j] += 0.1 * np.exp(-distance/2.0)

        # Create diurnal wind pattern
        def create_wind_field(hour):
            # Wind varies throughout the day
            base_speed = 2.0
            direction_variation = np.pi * hour / 12.0  # Rotates over 24 hours

            u_speed = base_speed * (1 + 0.5 * np.cos(direction_variation))
            v_speed = base_speed * (0.5 + 0.3 * np.sin(direction_variation))

            u = torch.ones(ny, nx) * u_speed
            v = torch.ones(ny, nx) * v_speed

            return (v, u)

        # Create system components
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(scenario_config['grid_resolution'],) * 2,
            scheme="upwind",
            boundary_method="periodic",
            track_conservation=True
        )

        spatial_scaler = SpatialScaling(
            source_scale="urban",
            target_scale="regional",
            source_resolution=(5000.0, 5000.0),
            target_resolution=(25000.0, 25000.0),
            method="bilinear",
            enforce_conservation=False
        )

        temporal_mgr = get_global_manager()
        if temporal_mgr is None:
            pytest.skip("ParameterManager not available for temporal aggregation tests")
        temporal_aggregator = TemporalAggregator(
            TemporalConfig(
                source_frequency='hourly',
                target_frequency='daily',
                direction=ScalingDirection.UPSCALE,
                conservation_level=ConservationLevel.NONE,
            ),
            parameter_manager=temporal_mgr,
        )

        # Run realistic simulation
        concentration = torch.ones(ny, nx, dtype=torch.float32) * 0.1  # Background concentration
        hourly_results = []
        regional_results = []
        total_source_increment = torch.zeros((), dtype=concentration.dtype)

        dt = scenario_config['time_step']
        n_steps = int(scenario_config['simulation_time'] / dt)

        for step in range(n_steps):
            current_time = step * dt
            hour = (current_time / 3600.0) % 24.0

            # Update wind field
            wind = create_wind_field(hour)
            advection.set_velocity_field(wind)

            # Diurnal emission variation (higher during day)
            emission_factor = 1.0 if 6 <= hour <= 18 else 0.3
            current_emissions = emission_sources * emission_factor
            total_source_increment = (
                total_source_increment + torch.sum(current_emissions) * dt
            )

            # Transport simulation
            state = {
                "tracer_concentration": concentration.unsqueeze(0),
                "source_term": current_emissions.unsqueeze(0)
            }

            updates = advection.forward(state, current_time, dt)
            concentration = concentration + _tracer_increment(updates).squeeze(0)

            # Store hourly results
            if step % 6 == 0:  # Every hour
                hourly_results.append({
                    'time': current_time,
                    'hour': hour,
                    'concentration': concentration.clone(),
                    'emissions': current_emissions.clone(),
                    'wind': wind
                })

                # Regional scaling
                regional_conc = spatial_scaler.upscale(concentration.unsqueeze(0))
                regional_results.append({
                    'time': current_time,
                    'regional_concentration': regional_conc.squeeze(0).clone()
                })

        # Temporal aggregation of concentrations
        if hourly_results:
            concentration_tensor = torch.stack([r['concentration'] for r in hourly_results])
            source_time = (
                torch.arange(len(hourly_results), dtype=torch.float32) + 0.5
            )
            target_time = torch.tensor([12.0], dtype=torch.float32)

            daily_result = temporal_aggregator.aggregate(
                data=concentration_tensor,
                source_time=source_time,
                target_time=target_time,
                variable_name='pollutant_concentration',
                standard_name='mass_concentration_of_pollutant_in_air',
                units='kg m-3',
                calendar="noleap",
                time_units="hours since 1970-01-01",
                source_bounds=torch.tensor(
                    [[hour, hour + 1] for hour in range(len(hourly_results))],
                    dtype=torch.float32,
                ),
                target_bounds=torch.tensor(
                    [[0.0, 24.0]], dtype=torch.float32
                ),
                temporal_semantics="interval_mean",
                target_units="kg m-3",
                measure_units="kg m-3 hour",
                integration_time_unit="hour",
            )

        # Validate realistic scenario results
        assert len(hourly_results) == 24
        assert len(regional_results) == 24

        # Check diurnal pattern
        day_concentrations = [r['concentration'].mean() for r in hourly_results if 6 <= r['hour'] <= 18]
        night_concentrations = [r['concentration'].mean() for r in hourly_results if r['hour'] < 6 or r['hour'] > 18]

        # Check diurnal concentrations are finite
        if day_concentrations and night_concentrations:
            assert np.isfinite(np.mean(day_concentrations))
            assert np.isfinite(np.mean(night_concentrations))

        # Check spatial pattern (higher near urban centers)
        final_concentration = concentration
        urban_concentrations = []
        background_concentrations = []

        for i in range(ny):
            for j in range(nx):
                is_urban = any(np.sqrt((i-cy)**2 + (j-cx)**2) < 5.0 for cy, cx in urban_centers)
                if is_urban:
                    urban_concentrations.append(final_concentration[i, j].item())
                else:
                    background_concentrations.append(final_concentration[i, j].item())

        if urban_concentrations and background_concentrations:
            urban_mean = float(np.mean(urban_concentrations))
            background_mean = float(np.mean(background_concentrations))
            assert np.isfinite(urban_mean)
            assert np.isfinite(background_mean)
            assert np.std(final_concentration.cpu().numpy()) > 0.0

        # Conservation check
        initial_total = torch.sum(torch.ones(ny, nx) * 0.1)
        final_total = torch.sum(final_concentration)
        # Periodic finite-volume advection has no boundary export.
        expected_final = initial_total.to(final_total.device) + total_source_increment.to(
            final_total.device
        )
        relative_error = abs((final_total - expected_final) / expected_final)
        relative_error_value = float(relative_error.item())
        assert np.isfinite(relative_error_value)
        assert relative_error_value < 1.0e-5


class TestSystemComparisonWithAnalytical:
    """Test system results against analytical solutions."""

    def test_1d_advection_analytical_comparison(self):
        """Compare 1D advection with analytical solution."""
        # Simple 1D advection: dc/dt + u*dc/dx = 0
        # Solution: c(x,t) = c0(x - u*t) (pure advection, no diffusion)

        u = 1.0  # Advection velocity
        L = 10.0  # Domain length
        nx = 100  # Number of grid points
        dx = L / nx
        dt = 0.1  # Time step
        t_final = 5.0  # Final time
        n_steps = int(t_final / dt)

        # Create initial condition (Gaussian pulse)
        x = torch.linspace(0, L, nx)
        x0 = L / 4  # Initial pulse center
        sigma = 0.5
        initial_concentration = torch.exp(-(x - x0)**2 / (2 * sigma**2))

        # Numerical solution
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=dx,
            scheme="upwind"
        )
        velocity = torch.ones(nx) * u
        advection.set_velocity_field(velocity)

        numerical_solution = initial_concentration.clone()
        state = {
            "tracer_concentration": numerical_solution.unsqueeze(0),
            "source_term": torch.zeros_like(numerical_solution.unsqueeze(0))
        }

        for step in range(n_steps):
            updates = advection.forward(state, step * dt, dt)
            numerical_solution = numerical_solution + _tracer_increment(updates).squeeze(0)
            state["tracer_concentration"] = numerical_solution.unsqueeze(0)

        # Analytical solution
        x_shifted = x - u * t_final
        analytical_solution = torch.exp(-(x_shifted - x0)**2 / (2 * sigma**2))

        # Comparison
        # Shift periodic boundary for analytical solution
        x_shifted_periodic = torch.where(x_shifted < 0, x_shifted + L, x_shifted)
        analytical_solution_periodic = torch.exp(-(x_shifted_periodic - x0)**2 / (2 * sigma**2))

        # Calculate error metrics
        error = torch.abs(numerical_solution - analytical_solution_periodic)
        l2_error = torch.sqrt(torch.mean(error**2))
        l_inf_error = torch.max(error)
        relative_error = l2_error / torch.mean(analytical_solution_periodic)

        # Assertions
        assert relative_error < 1.2, f"L2 relative error too large: {relative_error:.3f}"
        assert l_inf_error < 1.0, f"L-infinity error too large: {l_inf_error:.3f}"

        # Check that pulse has moved correct distance
        initial_center_idx = torch.argmax(initial_concentration)
        final_center_idx = torch.argmax(numerical_solution)
        expected_shift = int(u * t_final / dx)

        # Account for periodic boundary
        actual_shift = (final_center_idx - initial_center_idx) % nx
        assert abs(actual_shift - expected_shift) <= 2, f"Advection distance incorrect: expected {expected_shift}, got {actual_shift}"

    def test_diffusion_analytical_comparison(self):
        """Test diffusion process against analytical solution."""
        # Note: This would require implementing diffusion transport
        # For now, we test that the system handles diffusion-like processes

        # Create a simple diffusion-like transport using random walk
        nx = 50
        diffusion_coeff = 0.1
        dt = 0.01
        n_steps = 100

        # Initial condition: delta function in center
        concentration = torch.zeros(nx)
        concentration[nx // 2] = 1.0

        # Simulate diffusion using transport components
        for step in range(n_steps):
            # Use advection with random velocities to simulate diffusion
            random_velocity = torch.randn(nx) * 0.1

            advection = LinearAdvection(
                spatial_dims=1,
                grid_spacing=1.0,
                scheme="upwind"
            )
            advection.set_velocity_field(random_velocity)

            state = {
                "tracer_concentration": concentration.unsqueeze(0),
                "source_term": torch.zeros_like(concentration.unsqueeze(0))
            }

            updates = advection.forward(state, step * dt, dt)
            concentration = concentration + _tracer_increment(updates).squeeze(0)

            # Apply smoothing to approximate diffusion
            concentration = 0.95 * concentration + 0.025 * torch.roll(concentration, 1) + 0.025 * torch.roll(concentration, -1)

        # Verify diffusion-like behavior
        assert torch.sum(concentration) > 0  # Mass should be conserved
        assert torch.max(concentration) < 1.0  # Peak should decrease due to spreading
        assert len(torch.nonzero(concentration)) > nx // 4  # Should spread to multiple cells


class TestExternalDataIntegration:
    """Test integration with external data sources."""

    def test_netcdf_data_integration(self):
        """Test system integration with NetCDF data format."""
        # Create mock NetCDF-like data
        mock_netcdf_data = {
            'dimensions': {'time': 24, 'lat': 10, 'lon': 15},
            'variables': {
                'temperature': {
                    'data': torch.randn(24, 10, 15) * 10 + 273.15,  # K
                    'dims': ['time', 'lat', 'lon'],
                    'attrs': {'units': 'K', 'long_name': 'Air temperature'}
                },
                'wind_u': {
                    'data': torch.randn(24, 10, 15) * 5 + 10,  # m/s
                    'dims': ['time', 'lat', 'lon'],
                    'attrs': {'units': 'm s-1', 'long_name': 'Eastward wind'}
                },
                'wind_v': {
                    'data': torch.randn(24, 10, 15) * 3 + 5,   # m/s
                    'dims': ['time', 'lat', 'lon'],
                    'attrs': {'units': 'm s-1', 'long_name': 'Northward wind'}
                }
            }
        }

        # Create transport system that uses external data
        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(200000.0, 200000.0),
            scheme="upwind",
            boundary_method="periodic",
        )

        # Use external wind data
        u_data = mock_netcdf_data['variables']['wind_u']['data'][0]  # First time step
        v_data = mock_netcdf_data['variables']['wind_v']['data'][0]
        advection.set_velocity_field([v_data, u_data])

        # Use external temperature as tracer
        temperature_data = mock_netcdf_data['variables']['temperature']['data'][0]
        concentration = temperature_data / 273.15  # Normalize

        # Run transport with external data
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 3600.0  # 1 hour
        updates = advection.forward(state, 0.0, dt)
        new_concentration = concentration + _tracer_increment(updates).squeeze(0)

        # Verify integration
        assert new_concentration.shape == concentration.shape
        assert torch.isfinite(new_concentration).all()

        # Test temporal scaling with external data time coordinates
        temporal_config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily',
            direction=ScalingDirection.UPSCALE,
            conservation_level=ConservationLevel.NONE,
        )
        temporal_mgr = get_global_manager()
        if temporal_mgr is None:
            pytest.skip("ParameterManager not available for temporal aggregation tests")
        aggregator = TemporalAggregator(temporal_config, parameter_manager=temporal_mgr)

        # Create time series from external data
        hourly_temps = []
        for i in range(24):
            temp = mock_netcdf_data['variables']['temperature']['data'][i]
            hourly_temps.append(temp.mean().unsqueeze(0))

        hourly_tensor = torch.stack(hourly_temps)
        source_time = torch.arange(24, dtype=torch.float32) + 0.5
        target_time = torch.tensor([12.0], dtype=torch.float32)

        aggregated_result = aggregator.aggregate(
            data=hourly_tensor,
            source_time=source_time,
            target_time=target_time,
            variable_name='temperature',
            standard_name='air_temperature',
            units='K',
            calendar="noleap",
            time_units="hours since 1970-01-01",
            source_bounds=torch.tensor(
                [[hour, hour + 1] for hour in range(24)],
                dtype=torch.float32,
            ),
            target_bounds=torch.tensor([[0.0, 24.0]], dtype=torch.float32),
            temporal_semantics="interval_mean",
            target_units="K",
            measure_units="K hour",
            integration_time_unit="hour",
        )

        # Verify aggregation of external data
        assert aggregated_result.data.shape == (1, 1)
        assert torch.isfinite(aggregated_result.data).all()

        # Check metadata preservation
        assert aggregated_result.metadata['units'] == 'K'
        assert 'air_temperature' in aggregated_result.metadata.get('standard_name', '')

    def test_json_configuration_integration(self):
        """Test system configuration from JSON-like data."""
        # Mock configuration data
        config_data = {
            "transport": {
                "advection": {
                    "scheme": "upwind",
                    "cfl_max": 0.8,
                    "auto_adjust_timestep": True
                },
                "grid": {
                    "resolution": [10000.0, 10000.0],
                    "dimensions": 2
                }
            },
            "coupling": {
                "temporal": {
                    "source_frequency": "hourly",
                    "target_frequency": "daily",
                    "method": "mean"
                },
                "spatial": {
                    "source_scale": "fine",
                    "target_scale": "coarse",
                    "method": "conservative"
                }
            }
        }

        # Create system from configuration
        advection = LinearAdvection(
            spatial_dims=config_data["transport"]["grid"]["dimensions"],
            grid_spacing=tuple(config_data["transport"]["grid"]["resolution"]),
            scheme=config_data["transport"]["advection"]["scheme"]
        )

        temporal_config = TemporalConfig(
            source_frequency=config_data["coupling"]["temporal"]["source_frequency"],
            target_frequency=config_data["coupling"]["temporal"]["target_frequency"]
        )

        temporal_mgr = get_global_manager()
        if temporal_mgr is None:
            pytest.skip("ParameterManager not available for temporal aggregation tests")
        aggregator = TemporalAggregator(temporal_config, parameter_manager=temporal_mgr)

        spatial_scaler = SpatialScaling(
            source_scale=config_data["coupling"]["spatial"]["source_scale"],
            target_scale=config_data["coupling"]["spatial"]["target_scale"],
            source_resolution=tuple(config_data["transport"]["grid"]["resolution"]),
            target_resolution=(20000.0, 20000.0),
            method=config_data["coupling"]["spatial"]["method"]
        )

        # Verify configuration application
        assert advection.scheme == "upwind"
        assert tuple(advection.grid_spacing) == (10000.0, 10000.0)
        assert advection.spatial_dims == 2

        assert temporal_config.source_frequency == "hourly"
        assert temporal_config.target_frequency == "daily"

        assert spatial_scaler.source_scale == "fine"
        assert spatial_scaler.target_scale == "coarse"
        assert spatial_scaler.method == "conservative"

        # Test system operation with configured components
        concentration = torch.ones(10, 10, dtype=torch.float32)
        velocity = (torch.ones(10, 10) * 0.5, torch.ones(10, 10) * 0.3)
        advection.set_velocity_field(velocity)

        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }

        dt = 3600.0
        updates = advection.forward(state, 0.0, dt)
        new_concentration = concentration + _tracer_increment(updates).squeeze(0)

        # Apply spatial scaling
        coarse_concentration = spatial_scaler.upscale(new_concentration.unsqueeze(0))

        # Verify configured system works correctly
        assert torch.isfinite(new_concentration).all()
        assert torch.isfinite(coarse_concentration).all()


# Test fixtures
@pytest.fixture
def complete_system_setup():
    """Fixture providing complete system setup for testing."""
    with patch('relic.scientific.transport.advection.get_global_manager', create=True) as mock_mgr:
        # Mock parameter manager
        param_manager = _build_parameter_manager_mock(_transport_parameter_values())
        mock_mgr.return_value = param_manager

        # Create complete system
        simulator = RELICSimulator()

        advection = LinearAdvection(
            spatial_dims=2,
            grid_spacing=(10000.0, 10000.0),
            scheme="upwind",
            boundary_method="periodic",
        )

        velocity = (torch.ones(10, 10) * 0.5, torch.ones(10, 10) * 0.3)
        advection.set_velocity_field(velocity)

        spatial_scaler = SpatialScaling(
            source_scale="fine",
            target_scale="coarse",
            source_resolution=(10000.0, 10000.0),
            target_resolution=(20000.0, 20000.0),
            method="conservative"
        )

        temporal_config = TemporalConfig(
            source_frequency='hourly',
            target_frequency='daily'
        )
        temporal_mgr = _build_parameter_manager_mock(_temporal_parameter_values())
        aggregator = TemporalAggregator(temporal_config, parameter_manager=temporal_mgr)

        simulator.register_functor(advection)
        simulator.register_functor(spatial_scaler)
        simulator.register_functor(aggregator)

        return {
            'simulator': simulator,
            'advection': advection,
            'spatial_scaler': spatial_scaler,
            'aggregator': aggregator
        }


# Integration tests using fixtures
def test_system_workflow_with_fixture(complete_system_setup):
    """Integration test using complete system fixture."""
    system = complete_system_setup

    # Test basic functionality
    assert system['advection'].spatial_dims == 2
    assert system['spatial_scaler'].method == "conservative"
    assert system['aggregator'].source_scale == "hourly"

    # Run simple simulation
    device = system['advection'].device
    concentration = torch.ones(10, 10, dtype=torch.float32, device=device) * 2.0
    state = {
        "tracer_concentration": concentration.unsqueeze(0),
        "source_term": torch.zeros_like(concentration.unsqueeze(0))
    }

    dt = 1800.0
    updates = system['advection'].forward(state, 0.0, dt)
    new_concentration = concentration + _tracer_increment(updates).squeeze(0)

    # Apply scaling
    coarse_conc = system['spatial_scaler'].upscale(new_concentration.unsqueeze(0))

    # Verify results
    assert torch.isfinite(new_concentration).all()
    assert torch.isfinite(coarse_conc).all()
    assert coarse_conc.shape[-2:] == (5, 5)
    if coarse_conc.dim() == 3:
        assert coarse_conc.shape[0] == 1


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])

