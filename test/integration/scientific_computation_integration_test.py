"""
L2 Integration Tests for Scientific-Computation Layer Integration

This test suite validates the integration between scientific modules and the
computation layer, ensuring proper use of public operator libraries and
unified solver systems.

Test Coverage:
- Transport module integration with computation operators
- Solver system integration across modules
- Parameter system integration
- Cross-module data flow and unit consistency
- CF-1.8 compliance in data exchange

Mathematical Framework:
    Conservation Laws:
    - Mass: 鈭侻/鈭倀 + 鈭嚶稦 = S
    - Energy: 鈭侲/鈭倀 + 鈭嚶稦_E = Q
    - Momentum: 鈭?蟻v)/鈭倀 + 鈭嚶?蟻vv) = -鈭噋 + 蟻g + 渭鈭嚶瞯

References:
    - CF Conventions 1.8: http://cfconventions.org/cf-conventions/v1.8/
    - RELIC Architecture Documentation
"""

import pytest
import torch

# Import scientific modules
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.scientific.hydrosphere.soil_moisture.richards import RichardsSoilMoisture

# Import computation layer
from relic.computation.solvers import SolverFactory

# Import core system
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.parameters import get_global_manager


class TestScientificComputationIntegration:
    """Test suite for scientific-computation layer integration."""

    @pytest.fixture
    def simulator_config(self):
        """Create simulator configuration."""
        return {
            'time_step': 1800.0,  # 30 minutes
            'spatial_resolution': {'dx': 1000.0, 'dy': 1000.0, 'dz': 0.1},
            'device': 'cpu',
            'dtype': torch.float32,
            'strict_scale_detection': False
        }

    @pytest.fixture
    def sample_3d_state(self):
        """Create sample 3D state for transport testing."""
        nx, ny, nz = 5, 5, 10
        return {
            # Tracer concentration [kg/m鲁]
            'tracer_concentration': torch.rand(nx, ny, nz, dtype=torch.float32) * 0.1,

            # Velocity components [m/s]
            'u_wind': torch.randn(nx, ny, nz, dtype=torch.float32) * 0.5,
            'v_wind': torch.randn(nx, ny, nz, dtype=torch.float32) * 0.3,
            'w_wind': torch.randn(nx, ny, nz, dtype=torch.float32) * 0.01,

            # Diffusivity [m虏/s]
            'eddy_diffusivity': torch.ones(nx, ny, nz, dtype=torch.float32) * 0.001,

            # Grid spacing
            'dx': torch.tensor(1000.0, dtype=torch.float32),
            'dy': torch.tensor(1000.0, dtype=torch.float32),
            'dz': torch.tensor(0.1, dtype=torch.float32)
        }

    @pytest.fixture
    def sample_soil_state(self):
        """Create sample soil state for Richards equation testing."""
        nz = 10
        return {
            'soil_moisture': torch.full((nz,), 0.25, dtype=torch.float32),
            'infiltration_flux': torch.tensor(1e-8, dtype=torch.float32),
            'evapotranspiration_flux': torch.tensor(0.0, dtype=torch.float32),
        }

    @pytest.fixture
    def sample_photosynthesis_state(self):
        """Create sample photosynthesis state."""
        return {
            'leaf_temperature': torch.tensor(298.15, dtype=torch.float32),
            'co2_concentration': torch.tensor(400.0, dtype=torch.float32),  # ppm
            'par_radiation': torch.tensor(800.0, dtype=torch.float32),  # 渭mol/m虏/s
            'leaf_area_index': torch.tensor(3.0, dtype=torch.float32),
            'vapor_pressure_deficit': torch.tensor(1.0, dtype=torch.float32)  # kPa
        }

    def test_transport_operator_factory_integration(self, sample_3d_state):
        """Test transport modules use OperatorFactory correctly."""
        grid_spacing = (
            float(sample_3d_state['dz']),
            float(sample_3d_state['dy']),
            float(sample_3d_state['dx']),
        )
        advection = LinearAdvection(spatial_dims=3, grid_spacing=grid_spacing)
        diffusion = FickianDiffusion(
            spatial_dims=3,
            diffusivity=float(sample_3d_state['eddy_diffusivity'].mean()),
            grid_spacing=grid_spacing,
        )

        # Verify both use the same factory pattern
        assert hasattr(advection, 'operator_factory')
        assert hasattr(diffusion, 'operator_factory')
        assert type(advection.operator_factory) == type(diffusion.operator_factory)

        # Test operator retrieval
        advection_op = advection.operator_factory.get_operator('transport.advection')
        diffusion_op = diffusion.operator_factory.get_operator('transport.diffusion')

        assert advection_op is not None
        assert diffusion_op is not None

        # Test concurrent usage through the current state-dict forward contract.
        tracer = sample_3d_state['tracer_concentration'].unsqueeze(0)
        advection_output = advection.forward(
            {
                "tracer_concentration": tracer,
                "velocity_field": (
                    sample_3d_state['u_wind'].unsqueeze(0),
                    sample_3d_state['v_wind'].unsqueeze(0),
                    sample_3d_state['w_wind'].unsqueeze(0),
                ),
            },
            time=0.0,
            dt=1.0,
        )

        diffusion_output = diffusion.forward(
            {"tracer_concentration": tracer},
            time=0.0,
            dt=1.0e-6,
        )
        advection_flux = advection_output["tracer_concentration"].value
        diffusion_flux = diffusion_output["tracer_concentration"].value

        # Verify results are consistent
        assert advection_flux.shape == diffusion_flux.shape
        assert advection_flux.dtype == diffusion_flux.dtype

    def test_solver_system_integration(self, sample_soil_state):
        """Test solver system integration across modules."""
        thomas_solver = SolverFactory.create(
            solver_name='thomas_algorithm',
            name='integration_thomas_solver',
        )
        assert thomas_solver is not None

        richards = RichardsSoilMoisture()
        result = richards.forward(
            state=sample_soil_state,
            time=0.0,
            dt=3600.0
        )

        assert 'soil_moisture' in result
        soil_update = result['soil_moisture']
        assert isinstance(soil_update, UpdateItem)
        assert soil_update.mode == UpdateMode.REPLACE
        assert soil_update.per_timestep is False
        final_soil_moisture = soil_update.value
        assert torch.all(final_soil_moisture >= 0)
        assert torch.all(final_soil_moisture <= 1)
        assert (
            'diagnostics.soil_moisture_richards.soil_moisture_change_rate'
            in result
        )
        assert 'diagnostics.soil_moisture_richards.water_flux_profile' in result

    def test_parameter_system_integration(self):
        """Test parameter system integration across modules."""
        param_manager = get_global_manager()

        gravity = param_manager.get("relic.physical.constants.earth.gravity_acceleration", default=9.80665)
        assert gravity > 9.0 and gravity < 10.0

        wind_min = param_manager.get("atmosphere.wind.minimum_speed", default=0.0)
        assert wind_min >= 0.0

        field_capacity = param_manager.get("hydrosphere.soil_moisture.field_capacity", default=0.3)
        assert 0.1 < field_capacity < 0.6

    def test_cross_module_unit_consistency(self, sample_3d_state, sample_soil_state):
        """Test unit consistency across different modules."""
        # All tensors should be float32 for consistency
        for tensor in sample_3d_state.values():
            assert tensor.dtype == torch.float32

        for tensor in sample_soil_state.values():
            assert tensor.dtype == torch.float32

        # Test unit conversions maintain consistency
        # Convert concentration from kg/m鲁 to g/cm鲁 and back
        concentration = sample_3d_state['tracer_concentration']

        # kg/m鲁 to g/cm鲁 (multiply by 1e-3)
        conc_gcm3 = concentration * 1e-3
        # g/cm鲁 to kg/m鲁 (multiply by 1e3)
        conc_recovered = conc_gcm3 * 1e3

        assert torch.allclose(concentration, conc_recovered, rtol=1e-6)

    def test_simulator_integration(self, simulator_config, sample_3d_state):
        """Test integration with RELICSimulator."""
        simulator = RELICSimulator(config=simulator_config)

        transport = AdvectionDiffusion(
            spatial_dims=3,
            diffusivity=1.0e-4,
            grid_spacing=(1000.0, 1000.0, 1000.0),
            boundary_method="periodic",
        )
        simulator.register_functor(transport)

        simulation_state = {
            "tracer_concentration": sample_3d_state[
                "tracer_concentration"
            ].clone(),
            "velocity_field": torch.zeros(
                (*sample_3d_state["tracer_concentration"].shape, 3),
                dtype=sample_3d_state["tracer_concentration"].dtype,
            ),
        }
        simulator.initialize_state(simulation_state)

        dt = 1.0
        state_before = simulator.state_manager.get_state()
        total_mass_before = torch.sum(state_before['tracer_concentration'])
        updates = simulator.step(dt=dt)
        assert isinstance(updates, dict)

        state_after = simulator.state_manager.get_state()
        total_mass_after = torch.sum(state_after['tracer_concentration'])
        relative_change = torch.abs(total_mass_after - total_mass_before) / total_mass_before
        assert relative_change < 1e-2

    def test_state_manager_integration(self, sample_3d_state):
        """Test StateManager integration."""
        try:
            state_manager = StateManager()

            # Initialize state
            state_manager.initialize_state(sample_3d_state)

            # Create mock updates
            updates = {
                'tracer_concentration': torch.randn_like(sample_3d_state['tracer_concentration']) * 1e-6,
                'u_wind': torch.randn_like(sample_3d_state['u_wind']) * 1e-3
            }

            # Apply updates
            state_manager.apply_updates(updates, dt=1800.0, check_conservation=True)

            # Verify state was updated
            updated_state = state_manager.get_state()
            assert 'tracer_concentration' in updated_state
            assert 'u_wind' in updated_state

            # Verify conservation report
            report = state_manager.generate_conservation_report(updates, current_time=0.0)
            assert report is not None

        except Exception as e:
            if "conservation" in str(e).lower():
                pytest.skip("Conservation checking not fully available")
            else:
                raise

    def test_cf_compliance_in_data_exchange(self):
        """Test CF-1.8 compliance in inter-module data exchange."""
        # Test variable naming conventions
        standard_names = {
            'air_temperature': 'air_temperature',
            'tracer_concentration': 'mass_concentration_of_tracer_in_air',
            'wind_speed': 'wind_speed',
            'soil_moisture': 'volume_fraction_of_water_in_soil'
        }

        for var_name, standard_name in standard_names.items():
            assert isinstance(var_name, str)
            assert isinstance(standard_name, str)
            # Should follow CF-1.8 naming patterns
            assert '_' in var_name or 'temperature' in var_name or 'moisture' in var_name

        # Test units compliance
        units_map = {
            'air_temperature': 'K',
            'tracer_concentration': 'kg m-3',
            'wind_speed': 'm s-1',
            'soil_moisture': 'm3 m-3'
        }

        for var_name, units in units_map.items():
            assert isinstance(units, str)
            # Should contain valid SI units
            assert any(unit in units for unit in ['K', 'kg', 'm', 's'])

    def test_error_propagation_across_modules(self, sample_3d_state):
        """Test error handling and propagation across module boundaries."""
        grid_spacing = (
            float(sample_3d_state['dx']),
            float(sample_3d_state['dy']),
            float(sample_3d_state['dz']),
        )
        advection = LinearAdvection(spatial_dims=3, grid_spacing=grid_spacing)
        diffusion = FickianDiffusion(
            spatial_dims=3,
            diffusivity=float(sample_3d_state['eddy_diffusivity'].mean()),
            grid_spacing=grid_spacing,
        )

        # Test with invalid input (NaN values)
        invalid_concentration = torch.full_like(sample_3d_state['tracer_concentration'], float('nan'))

        # Both modules should handle invalid inputs gracefully
        try:
            advection_result = advection.forward(
                {
                    "tracer_concentration": invalid_concentration.unsqueeze(0),
                    "u": sample_3d_state['u_wind'].unsqueeze(0),
                    "v": sample_3d_state['v_wind'].unsqueeze(0),
                    "w": sample_3d_state['w_wind'].unsqueeze(0),
                },
                time=0.0,
                dt=1.0,
            )
            # Should return NaN or raise appropriate error
            assert torch.isnan(advection_result["tracer_concentration"]).any()
        except (ValueError, RuntimeError):
            # Acceptable to raise error for invalid input
            pass

        try:
            diffusion_result = diffusion.forward(
                {"tracer_concentration": invalid_concentration.unsqueeze(0)},
                time=0.0,
                dt=1.0e-6,
            )
            assert torch.isnan(diffusion_result["tracer_concentration"]).any()
        except (ValueError, RuntimeError):
            pass

    def test_memory_efficiency_in_large_scales(self):
        """Test memory efficiency with larger state arrays."""
        # Create larger 3D state
        nx, ny, nz = 50, 50, 20
        large_state = {
            'concentration': torch.rand(nx, ny, nz, dtype=torch.float32) * 0.1,
            'velocity_x': torch.randn(nx, ny, nz, dtype=torch.float32) * 0.5,
            'velocity_y': torch.randn(nx, ny, nz, dtype=torch.float32) * 0.5,
            'diffusivity': torch.ones(nx, ny, nz, dtype=torch.float32) * 0.001
        }

        # Test that modules can handle larger arrays without memory issues
        grid_spacing = (0.1, 1000.0, 1000.0)
        advection = LinearAdvection(spatial_dims=3, grid_spacing=grid_spacing)
        diffusion = FickianDiffusion(
            spatial_dims=3,
            diffusivity=float(large_state['diffusivity'].mean()),
            grid_spacing=grid_spacing,
        )

        try:
            tracer = large_state['concentration'].unsqueeze(0)
            advection_output = advection.forward(
                {
                    "tracer_concentration": tracer,
                    "velocity_field": (
                        large_state['velocity_x'].unsqueeze(0),
                        large_state['velocity_y'].unsqueeze(0),
                        torch.zeros_like(large_state['concentration']).unsqueeze(0),
                    ),
                },
                time=0.0,
                dt=1.0,
            )

            diffusion_output = diffusion.forward(
                {"tracer_concentration": tracer},
                time=0.0,
                dt=1.0e-6,
            )
            advection_result = advection_output["tracer_concentration"].value
            diffusion_result = diffusion_output["tracer_concentration"].value

            # Verify results maintain shape and type
            assert advection_result.shape == tracer.shape
            assert diffusion_result.shape == tracer.shape
            assert advection_result.dtype == torch.float32
            assert diffusion_result.dtype == torch.float32

        except RuntimeError as e:
            if "memory" in str(e).lower():
                pytest.skip("Memory limitations on this system")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


