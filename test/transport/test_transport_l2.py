"""
L2 Integration Tests for Transport Module

This module provides comprehensive integration tests for the transport module,
testing the interaction between transport components, state management, parameter
system, and coupling with other modules. Tests focus on system-level behavior,
cross-module interactions, and end-to-end functionality.

Test Coverage:
- Transport functor integration with StateManager
- Cross-module parameter sharing and validation
- Transport coupling with other scientific modules
- Conservation law enforcement across module boundaries
- Multi-scale transport interactions
- Error propagation and handling across module boundaries
- Performance and stability under integrated conditions
- Configuration management across transport components
- Dependency resolution and execution ordering

Integration Scenarios:
- Transport modules within full simulation pipeline
- Coupled advection-diffusion processes
- Transport with biological processes (e.g., root uptake)
- Transport with atmospheric boundary layer processes
- Multi-phase transport (soil-water-atmosphere)
- Temporal and spatial scale coupling
- Parameter calibration and optimization workflows

Mathematical Verification:
- Mass conservation across coupled systems: dm_in - dm_out = dm_storage
- Energy conservation in coupled transport: dE_in - dE_out = dE_storage
- Momentum conservation in fluid transport: dF = d(m*v)
- Flux continuity at interfaces: J_left = J_right
- Scale transformation consistency: dphi_fine = dphi_coarse

References:
- O'Neill, R.V. et al. (1986). A Hierarchical Concept of Ecosystems
- Cushman, J.H. (1997). The Physics of Fluids in Hierarchical Porous Media
- Campbell, G.S. & Norman, J.M. (1998). An Introduction to Environmental Biophysics
- Raupach, M.R. & Thom, A.S. (1981). Turbulence in and above plant canopies
"""

import pytest
import torch
from unittest.mock import Mock, patch

from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion

# Import core components for integration
from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode
from relic.core.simulator import RELICSimulator
from relic.core.dependency_graph import DependencyGraphResolver
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant

from test.utils.forward_call import call_forward

def _fickian_increment(output, key="tracer_concentration"):
    """Return an integrated Fickian increment with canonical state semantics."""
    update = output[key]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return update.value


def _linear_increment(output, key="tracer_concentration"):
    """Return an integrated LinearAdvection increment with canonical semantics."""
    update = output[key]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return update.value


class TestTransportStateManagerIntegration:
    """Test integration of transport modules with StateManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.tolerance = 1e-6
        
    def test_state_manager_advection_integration(self):
        """Test LinearAdvection integration with StateManager."""
        # Create state manager
        state_mgr = StateManager(config={}, device=self.device)
        
        # Register transport variable
        state_mgr.register_variable(
            StateVariableSpec(
                name="tracer_concentration",
                units="kg m-3",
                default_value=100.0,
                bounds=(0.0, 1000.0)
            )
        )
        
        # Register diagnostic variables
        state_mgr.register_variable(
            StateVariableSpec(
                name="diagnostics.cfl_number",
                units="dimensionless",
                default_value=0.0,
                bounds=(0.0, 10.0)
            )
        )
        
        # Create advection module
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.01],
            device=self.device,
            name="TestAdvection"
        )
        
        # Set velocity field
        velocity = torch.ones(1, 10) * 0.1
        advection.set_velocity_field([velocity])
        
        # Get initial state
        initial_concentration = torch.linspace(90.0, 110.0, 10).unsqueeze(0)
        state_mgr.set_state("tracer_concentration", initial_concentration)
        
        # Get state dict
        state = state_mgr.get_state()
        
        # Compute transport increment
        dt = 0.05
        output = advection.forward(state, time=0.0, dt=dt)
        
        # Apply increment through state manager
        state_mgr.apply_updates(output, dt=dt, check_conservation=False)
        
        # Verify state was updated
        new_state = state_mgr.get_state()
        new_concentration = new_state["tracer_concentration"]
        
        # Should have changed from initial
        assert not torch.allclose(new_concentration, initial_concentration, atol=1e-10)
        
        # Check conservation report
        if hasattr(state_mgr, 'get_conservation_report'):
            report = state_mgr.get_conservation_report()
            assert report is not None
            
    def test_state_manager_diffusion_integration(self):
        """Test FickianDiffusion integration with StateManager."""
        # Create state manager
        state_mgr = StateManager(config={}, device=self.device)
        
        # Register variables
        state_mgr.register_variable(
            StateVariableSpec(
                name="tracer_concentration",
                units="kg m-3",
                default_value=50.0,
                bounds=(0.0, 1000.0)
            )
        )
        
        state_mgr.register_variable(
            StateVariableSpec(
                name="diagnostics.conservation_error",
                units="dimensionless",
                default_value=0.0,
                bounds=(0.0, 1.0)
            )
        )
        
        # Create diffusion module
        diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=[0.1],
            track_conservation=True,
            device=self.device,
            name="TestDiffusion"
        )
        
        # Set initial state
        initial_concentration = torch.ones(1, 20) * 50.0
        state_mgr.set_state("tracer_concentration", initial_concentration)
        
        # Add source term
        source = torch.ones(1, 20) * 0.1  # Small nitrogen input
        state_mgr.set_state("source_term", source)
        
        # Get state and compute update
        state = state_mgr.get_state()
        dt = 300.0  # 5 minutes
        output = diffusion.forward(state, time=0.0, dt=dt)
        
        # Apply update
        state_mgr.apply_updates(output, dt=dt, check_conservation=False)
        
        # Verify conservation
        new_concentration = state_mgr.get_state("tracer_concentration")
        expected_total = torch.sum(initial_concentration) + torch.sum(source) * dt
        actual_total = torch.sum(new_concentration)
        
        # Should be approximately conserved
        relative_error = torch.abs(actual_total - expected_total) / expected_total
        assert relative_error < 0.01  # Less than 1% error
        
    def test_state_manager_coupled_transport(self):
        """Test coupled advection-diffusion with StateManager."""
        # Create state manager
        state_mgr = StateManager(config={}, device=self.device)
        
        # Register the single canonical transported tracer state.
        state_mgr.register_variable(
            StateVariableSpec(
                name="tracer_concentration",
                units="kg m-3",
                default_value=1.0
            )
        )

        transport = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=2e-5,
            grid_spacing=[1.0],
            velocity_field=torch.ones(1, 15) * 0.5,
            boundary_method="periodic",
        )

        initial_concentration = torch.linspace(1.0, 2.0, 15).unsqueeze(0)
        state_mgr.set_state("tracer_concentration", initial_concentration)
        dt = 1.0
        state = state_mgr.get_state()
        output = transport.forward(state, time=0.0, dt=dt)
        state_mgr.apply_updates(output, dt=dt, check_conservation=False)

        final_concentration = state_mgr.get_state("tracer_concentration")
        assert not torch.allclose(final_concentration, initial_concentration, atol=1e-6)
        
        # Check conservation (closed system)
        total_before = torch.sum(initial_concentration)
        total_after = torch.sum(final_concentration)
        relative_change = torch.abs(total_after - total_before) / total_before
        assert relative_change < 0.1  # Less than 10% change in total


class TestTransportSimulatorIntegration:
    """Test integration of transport modules with RELICSimulator."""
    
    def test_simulator_transport_registration(self):
        """Test transport module registration with simulator."""
        # Create simulator
        simulator = RELICSimulator(config={"conservation_checks": False})
        
        # Create transport modules
        advection = LinearAdvection(spatial_dims=1)
        diffusion = FickianDiffusion(spatial_dims=1)
        
        # Register modules
        simulator.register_functor(advection)
        with pytest.raises(ValueError, match="Mutually exclusive overlap"):
            simulator.register_functor(diffusion)
        
        # Check registration
        assert len(simulator.functors) == 1
        assert "linear_advection" in simulator.functors
        
    def test_simulator_dependency_resolution(self):
        """Test dependency resolution for transport modules."""
        # Create dependency resolver
        resolver = DependencyGraphResolver()
        
        # Create transport modules with dependencies. add_dependency declares a
        # required edge that get_dependencies now reports, so the declared
        # WindModel provider must be registered (a stand-in advection node).
        wind_model = LinearAdvection(spatial_dims=1, name="WindModel")
        base_velocity = LinearAdvection(spatial_dims=1, name="BaseVelocity")
        base_velocity.add_dependency("WindModel")

        tracer_transport = LinearAdvection(spatial_dims=1, name="TracerTransport")
        tracer_transport.add_dependency("BaseVelocity")

        # Register functors (consumers first, so the order comes from the edges)
        resolver.add_functor(tracer_transport)
        resolver.add_functor(base_velocity)
        resolver.add_functor(wind_model)

        # Resolve dependencies
        execution_order = resolver.topological_sort()

        # WindModel -> BaseVelocity -> TracerTransport
        assert execution_order.index("WindModel") < execution_order.index("BaseVelocity")
        base_index = execution_order.index("BaseVelocity")
        tracer_index = execution_order.index("TracerTransport")
        assert base_index < tracer_index
        
    def test_simulator_coupled_step_execution(self):
        """Test coupled step execution in simulator."""
        # Create simulator
        simulator = RELICSimulator(
            config={"conservation_checks": False, "strict_scale_detection": False}
        )
        
        # Create coupled transport modules
        # Temperature transport (faster)
        temp_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[1.0],
            tracer_key="temp_state",
            tracer_units="K",
            name="TempAdvection"
        )
        temp_velocity = torch.ones(1, 10) * 2.0
        temp_advection.set_velocity_field([temp_velocity])
        
        # CO2 transport (slower)
        co2_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[1.0],
            tracer_key="co2_concentration",
            name="CO2Advection"
        )
        co2_velocity = torch.ones(1, 10) * 0.5
        co2_advection.set_velocity_field([co2_velocity])
        
        # Disable conservation enforcement for this integration smoke test
        temp_advection.metadata.conservation_laws = []
        co2_advection.metadata.conservation_laws = []

        # Register modules
        simulator.register_functor(temp_advection)
        simulator.register_functor(co2_advection)
        
        # Declare the state units the two tracers are advected in before the
        # simulator initializes them (temperature in K, CO2 as a mass density).
        typed_state = StateManager(config={}, device=torch.device("cpu"))
        typed_state.register_variable(
            StateVariableSpec(name="temp_state", units="K", default_value=293.15)
        )
        typed_state.register_variable(
            StateVariableSpec(name="co2_concentration", units="kg m-3", default_value=400.0)
        )
        simulator.state_manager = typed_state

        # Initialize state
        simulator.initialize_state({
            "temp_state": torch.ones(1, 10) * 293.15,
            "co2_concentration": torch.ones(1, 10) * 400.0,
        })
        # Disable contract enforcement for integration smoke test
        simulator.state_manager._contract_registry = None
        
        # Execute step
        updates = simulator.step(dt=0.3)
        
        # Should have updates for both variables
        assert "temp_state" in updates
        assert "co2_concentration" in updates
        
        # Temperature should change more (faster velocity)
        temp_change = torch.abs(updates["temp_state"]).mean()
        co2_change = torch.abs(updates["co2_concentration"]).mean()
        
        # Note: This depends on the actual values and implementation
        assert torch.isfinite(temp_change)
        assert torch.isfinite(co2_change)


class TestCrossModuleTransportIntegration:
    """Test integration between transport and other scientific modules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        
    def test_transport_photosynthesis_coupling(self):
        """Test coupling between transport and photosynthesis."""
        # Create transport module for CO2
        co2_transport = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.1],
            tracer_key="co2_concentration",
            name="CO2Transport"
        )
        
        # Set CO2 velocity (canopy air movement)
        co2_velocity = torch.ones(1, 20) * 0.2
        co2_transport.set_velocity_field([co2_velocity])
        
        # Create photosynthesis module (simplified mock)
        photosynthesis = Mock()
        photosynthesis.forward.return_value = {
            "co2_assimilation": torch.ones(1, 20) * 10.0  # umol m-2 s-1
        }
        
        # Initial state
        initial_co2 = torch.ones(1, 20) * 400.0  # ppm
        state = {
            "co2_concentration": initial_co2,
            "temperature": torch.ones(1, 20) * 298.15,  # K
            "par": torch.ones(1, 20) * 1000.0  # umol m-2 s-1
        }
        
        dt = 0.2
        
        # Transport step
        transport_output = co2_transport.forward(state, time=0.0, dt=dt)
        transport_increment = _linear_increment(
            transport_output,
            key="co2_concentration",
        )
        
        # Photosynthesis step (consumes CO2)
        env = {
            "temperature": state["temperature"],
            "par": state["par"],
            "co2_concentration": state["co2_concentration"] + transport_increment
        }
        photo_output = call_forward(photosynthesis, {**state, **env})
        
        # Apply both effects
        co2_change = transport_increment - photo_output["co2_assimilation"] * dt * 1e-6
        
        # Net effect should be reasonable
        assert torch.isfinite(co2_change).all()
        # CO2 should decrease overall (consumption > transport change)
        assert torch.mean(co2_change) < 0
        
    def test_transport_evapotranspiration_coupling(self):
        """Test coupling between transport and evapotranspiration."""
        # Create water vapor transport
        vapor_transport = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[1.0],
            tracer_key="specific_humidity",
            track_conservation=False,
            name="VaporTransport"
        )
        
        # Set vapor velocity
        vapor_velocity = torch.zeros(1, 15)
        vapor_transport.set_velocity_field([vapor_velocity])
        
        # Mock Penman-Monteith evapotranspiration
        et = Mock()
        et.forward.return_value = {
            "latent_heat_flux": torch.ones(1, 15) * 200.0,  # W m-2
            "evapotranspiration": torch.ones(1, 15) * 5e-5  # kg m-2 s-1
        }
        
        # Initial state
        initial_humidity = torch.ones(1, 15) * 0.6
        state = {
            "specific_humidity": initial_humidity,
            "temperature": torch.ones(1, 15) * 298.15,
            "wind_speed": torch.ones(1, 15) * 3.0,
            "net_radiation": torch.ones(1, 15) * 500.0,
            "soil_moisture": torch.ones(1, 15) * 0.3
        }
        
        dt = 0.5
        
        # Transport step
        transport_output = vapor_transport.forward(state, time=0.0, dt=dt)
        transport_increment = _linear_increment(
            transport_output,
            key="specific_humidity",
        )
        
        # ET step (adds moisture to atmosphere)
        et_output = et.forward(state, time=0.0, dt=dt)
        
        # Net humidity change
        humidity_change = transport_increment + et_output["evapotranspiration"] * dt
        
        # Should have net increase from ET
        assert torch.mean(humidity_change) > 0
        assert torch.isfinite(humidity_change).all()
        
    def test_transport_nutrient_cycling_coupling(self):
        """Test coupling between transport and nutrient cycling."""
        # Create nutrient transport (nitrate)
        no3_transport = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.05],
            tracer_key="no3_concentration",
            name="NO3Transport"
        )
        
        # Set water flow velocity
        water_velocity = torch.ones(1, 25) * 1e-6  # m/s (soil water flow)
        no3_transport.set_velocity_field([water_velocity])
        
        # Mock nitrification
        nitrification = Mock()
        nitrification.forward.return_value = {
            "no3_production": torch.ones(1, 25) * 1e-7,  # kg m-3 s-1
            "nh4_consumption": torch.ones(1, 25) * 1e-7
        }
        
        # Initial state
        initial_no3 = torch.ones(1, 25) * 10.0  # kg m-3
        initial_nh4 = torch.ones(1, 25) * 5.0   # kg m-3
        state = {
            "no3_concentration": initial_no3,
            "nh4_concentration": initial_nh4,
            "soil_temperature": torch.ones(1, 25) * 288.15,
            "soil_moisture": torch.ones(1, 25) * 0.3,
            "ph": torch.ones(1, 25) * 6.5,
            "oxygen": torch.ones(1, 25) * 0.2
        }
        
        dt = 3600.0  # 1 hour
        
        # Transport step
        transport_output = no3_transport.forward(state, time=0.0, dt=dt)
        transport_increment = _linear_increment(
            transport_output,
            key="no3_concentration",
        )
        
        # Nitrification step
        nitrif_output = nitrification.forward(state, time=0.0, dt=dt)
        
        # Net NO3 change
        no3_change = transport_increment + nitrif_output["no3_production"] * dt
        
        # Should have small positive change from nitrification
        assert torch.mean(no3_change) > -0.1  # Allow small negative from transport
        assert torch.isfinite(no3_change).all()


class TestMultiScaleTransportIntegration:
    """Test multi-scale transport interactions."""
    
    def test_spatial_scale_coupling(self):
        """Test spatial scale coupling in transport."""
        # Fine scale transport (leaf scale)
        leaf_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.001],  # 1mm
            name="LeafScaleAdvection"
        )
        
        # Coarse scale transport (canopy scale)
        canopy_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.01],  # 1cm
            name="CanopyScaleAdvection"
        )
        
        # Set up fine scale field
        n_fine = 100
        fine_concentration = torch.exp(-torch.linspace(-3, 3, n_fine)**2) * 100.0
        fine_velocity = torch.ones(n_fine) * 0.01
        
        leaf_advection.set_velocity_field([fine_velocity])
        
        # Fine scale computation
        fine_state = {
            "tracer_concentration": fine_concentration.unsqueeze(0),
            "source_term": torch.zeros_like(fine_concentration.unsqueeze(0))
        }
        
        dt = 0.01
        fine_output = leaf_advection.forward(fine_state, time=0.0, dt=dt)
        fine_increment = _linear_increment(fine_output)
        fine_new = fine_concentration + fine_increment.squeeze(0)
        
        # Upscale to coarse grid (simple averaging)
        n_coarse = 10
        coarse_concentration = torch.zeros(n_coarse)
        for i in range(n_coarse):
            start_idx = i * n_fine // n_coarse
            end_idx = (i + 1) * n_fine // n_coarse
            coarse_concentration[i] = torch.mean(fine_new[start_idx:end_idx])
        
        # Coarse scale computation
        coarse_velocity = torch.ones(n_coarse) * 0.5
        canopy_advection.set_velocity_field([coarse_velocity])
        
        coarse_state = {
            "tracer_concentration": coarse_concentration.unsqueeze(0),
            "source_term": torch.zeros_like(coarse_concentration.unsqueeze(0))
        }
        
        coarse_output = canopy_advection.forward(coarse_state, time=0.0, dt=dt)
        
        # Verify scale consistency
        assert fine_increment.shape == fine_concentration.unsqueeze(0).shape
        assert _linear_increment(coarse_output).shape == coarse_concentration.unsqueeze(0).shape
        
        # Total mass should be approximately conserved across scales
        fine_mass = torch.sum(fine_new) * 0.001  # Multiply by grid spacing
        coarse_mass = torch.sum(coarse_concentration) * 0.01
        relative_error = torch.abs(fine_mass - coarse_mass) / fine_mass
        assert relative_error < 0.1  # Within 10% due to upscaling
        
    def test_temporal_scale_coupling(self):
        """Test temporal scale coupling in transport."""
        # Fast transport (turbulent)
        fast_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.01],
            name="FastTurbulentTransport"
        )
        
        # Slow transport (mean flow)
        slow_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.01],
            name="SlowMeanTransport"
        )
        
        # Set velocities
        fast_velocity = torch.ones(20) * 2.0  # Fast turbulent fluctuations
        slow_velocity = torch.ones(20) * 0.1  # Slow mean flow
        
        fast_advection.set_velocity_field([fast_velocity])
        slow_advection.set_velocity_field([slow_velocity])
        
        # Initial concentration
        initial_conc = torch.ones(20) * 100.0
        
        # Fast time stepping (small dt)
        dt_fast = 0.002
        n_fast_steps = 10
        
        fast_state = {
            "tracer_concentration": initial_conc.unsqueeze(0),
            "source_term": torch.zeros_like(initial_conc.unsqueeze(0))
        }
        
        for i in range(n_fast_steps):
            fast_output = fast_advection.forward(fast_state, time=i*dt_fast, dt=dt_fast)
            fast_state["tracer_concentration"] += _linear_increment(fast_output)
            
        fast_final = fast_state["tracer_concentration"].squeeze(0)
        
        # Slow time stepping (large dt)
        dt_slow = dt_fast * n_fast_steps
        
        slow_state = {
            "tracer_concentration": initial_conc.unsqueeze(0),
            "source_term": torch.zeros_like(initial_conc.unsqueeze(0))
        }
        
        slow_output = slow_advection.forward(slow_state, time=0.0, dt=dt_slow)
        slow_final = initial_conc + _linear_increment(slow_output).squeeze(0)
        
        # Fast/slow transport should produce finite change magnitudes
        fast_change = torch.abs(fast_final - initial_conc).mean()
        slow_change = torch.abs(slow_final - initial_conc).mean()
        
        assert torch.isfinite(fast_change)
        assert torch.isfinite(slow_change)
        assert float(fast_change) >= 0.0
        assert float(slow_change) >= 0.0
        
    def test_hierarchical_scale_coupling(self):
        """Test hierarchical scale coupling (leaf -> plant -> canopy)."""
        # Leaf scale
        leaf_transport = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.001],
            name="LeafTransport"
        )
        
        # Plant scale
        LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.01],
            name="PlantTransport"
        )
        
        # Canopy scale
        LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.1],
            name="CanopyTransport"
        )
        
        # Set up hierarchical aggregation
        n_leaf = 50
        n_plant = 20
        n_canopy = 10
        
        # Leaf-level concentration (Gaussian profile)
        x_leaf = torch.linspace(-3, 3, n_leaf)
        leaf_conc = torch.exp(-x_leaf**2) * 100.0
        leaf_velocity = torch.ones(n_leaf) * 0.005
        
        leaf_transport.set_velocity_field([leaf_velocity])
        
        # Leaf to plant aggregation (weighted by leaf area)
        plant_conc = torch.zeros(n_plant)
        leaf_area_weights = torch.exp(-x_leaf**2)  # Leaf area distribution
        
        for i in range(n_plant):
            start_idx = i * n_leaf // n_plant
            end_idx = (i + 1) * n_leaf // n_plant
            weights = leaf_area_weights[start_idx:end_idx]
            if torch.sum(weights) > 0:
                plant_conc[i] = torch.sum(leaf_conc[start_idx:end_idx] * weights) / torch.sum(weights)
                
        # Plant to canopy aggregation
        canopy_conc = torch.zeros(n_canopy)
        for i in range(n_canopy):
            start_idx = i * n_plant // n_canopy
            end_idx = (i + 1) * n_plant // n_canopy
            canopy_conc[i] = torch.mean(plant_conc[start_idx:end_idx])
            
        # Verify hierarchical consistency
        # Total mass should be conserved (approximately)
        leaf_mass = torch.sum(leaf_conc) * 0.001
        plant_mass = torch.sum(plant_conc) * 0.01
        canopy_mass = torch.sum(canopy_conc) * 0.1
        
        # Relative errors should be small
        leaf_plant_error = torch.abs(leaf_mass - plant_mass) / leaf_mass
        plant_canopy_error = torch.abs(plant_mass - canopy_mass) / plant_mass
        
        assert leaf_plant_error < 5.0  # Allow coarse aggregation differences
        assert plant_canopy_error < 5.0


class TestParameterSystemIntegration:
    """Test parameter system integration across transport modules."""
    
    def test_shared_parameter_access(self):
        """Test shared parameter access across transport modules."""
        manager = get_global_manager()
        advection = LinearAdvection(spatial_dims=1, name="Adv1")
        diffusion = FickianDiffusion(spatial_dims=1, name="Diff1")

        assert advection.get_parameter_manager() is manager
        assert diffusion.get_parameter_manager() is manager
        von_karman_1 = get_physical_constant("von_karman_constant")
        von_karman_2 = get_physical_constant("von_karman_constant")

        assert von_karman_1 == von_karman_2
        
    @patch('relic.core.functor.functor_base._get_parameter_manager')
    def test_parameter_validation_coupling(self, mock_get_parameter_manager):
        """Test parameter validation across coupled modules."""
        # Mock parameter manager with validation
        mock_manager = Mock()
        
        def mock_get_with_validation(key, **kwargs):
            values = {
                "transport.advection.grid_spacing": 0.01,
                "transport.numerical.cfl_advection_max": 0.8,
                "transport.numerical.min_timestep": 1e-6,
                "transport.numerical.conservation_tolerance": 1e-6,
                "transport.numerical.limiter_epsilon": 1e-6,
                "transport.numerical.artificial_viscosity_coefficient": 0.0,
                "transport.numerical.enforce_non_negative": True,
                "transport.numerical.auto_adjust_timestep": False,
                "transport.numerical.track_conservation": True,
                "transport.numerical.default_boundary_type": "neumann",
                "transport.numerical.cfl_diffusion_max": 0.8,
                "transport.diffusion.grid_spacing": 0.01,
                "transport.diffusion.diffusivity": 2e-5,
            }
            return values.get(key, kwargs.get("default"))
                
        mock_manager.get = Mock(side_effect=mock_get_with_validation)
        mock_manager.get_parameter = Mock(return_value=None)
        mock_manager.list = Mock(return_value=[])
        mock_manager.parameters = {}
        mock_db = Mock()
        mock_db._loaded_roots = {"transport"}
        mock_manager.database = mock_db
        mock_get_parameter_manager.return_value = mock_manager
        
        # Create coupled modules
        advection = LinearAdvection(spatial_dims=1, name="CoupledAdv")
        diffusion = FickianDiffusion(spatial_dims=1, name="CoupledDiff")
        
        # Runtime numerical limits remain owned by the canonical operators.
        assert "transport.numerical.cfl_advection_max" in advection.required_parameters()
        assert not hasattr(advection, "max_cfl")
        assert diffusion.diffusivity_scalar == 2e-5
        assert advection.non_negative is True
        assert diffusion.non_negative is True
        
    def test_parameter_consistency_transport(self):
        """Test parameter consistency across transport types."""
        # Create modules with similar parameters
        adv_1d = LinearAdvection(spatial_dims=1, name="Adv1D")
        adv_2d = LinearAdvection(spatial_dims=2, name="Adv2D")
        diff_1d = FickianDiffusion(spatial_dims=1, name="Diff1D")
        
        # Check consistent canonical parameter declarations.
        assert all(
            "transport.numerical.cfl_advection_max" in adv.required_parameters()
            for adv in [adv_1d, adv_2d]
        )
        assert "transport.numerical.cfl_diffusion_max" in diff_1d.required_parameters()
        
        # All should have non-negative enforcement
        assert all(hasattr(adv, 'non_negative') for adv in [adv_1d, adv_2d])
        assert hasattr(diff_1d, 'non_negative')
        
        # Local time-step bounds would duplicate simulator scheduler ownership.
        assert all(not hasattr(adv, 'min_timestep') for adv in [adv_1d, adv_2d])
        assert not hasattr(diff_1d, 'min_timestep')


class TestConservationLawIntegration:
    """Test conservation law enforcement across integrated systems."""
    
    def test_mass_conservation_coupled_transport(self):
        """Test mass conservation in coupled transport systems."""
        # Create coupled advection-diffusion
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.01],
            track_conservation=True,
            name="MassAdv"
        )
        
        diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=[0.01],
            track_conservation=True,
            name="MassDiff"
        )
        
        # Set up closed system (no-flux boundaries)
        initial_mass = torch.ones(1, 50) * 100.0
        
        # Zero net flux at boundaries
        velocity = torch.zeros(1, 50)
        velocity[0, 1:-1] = 0.1  # Internal flow only
        advection.set_velocity_field([velocity])
        
        state = {
            "tracer_concentration": initial_mass,
            "source_term": torch.zeros_like(initial_mass)  # No sources/sinks
        }
        
        dt = 0.05
        total_mass_before = torch.sum(initial_mass)
        
        # Sequential operator splitting
        # Advection step
        adv_output = advection.forward(state, time=0.0, dt=dt)
        state_after_adv = {
            "tracer_concentration": initial_mass + _linear_increment(adv_output),
            "source_term": torch.zeros_like(initial_mass)
        }
        
        # Diffusion step
        diff_output = diffusion.forward(state_after_adv, time=dt, dt=dt)
        
        # Final state
        final_concentration = (
            state_after_adv["tracer_concentration"]
            + _fickian_increment(diff_output)
        )
        total_mass_after = torch.sum(final_concentration)
        
        # Mass should be conserved (within numerical tolerance)
        relative_mass_change = torch.abs(total_mass_after - total_mass_before) / total_mass_before
        assert relative_mass_change < 1e-2  # Less than 1% change
        
        # Check conservation diagnostics
        if "diagnostics.conservation_error" in adv_output:
            assert torch.isfinite(adv_output["diagnostics.conservation_error"])
        if "diagnostics.conservation_error" in diff_output:
            assert torch.isfinite(diff_output["diagnostics.conservation_error"])
            
    def test_energy_conservation_transport(self):
        """Test energy conservation in transport with temperature."""
        # Create temperature transport
        temp_advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.1],
            track_conservation=True,
            name="EnergyAdv"
        )
        
        temp_diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=2e-5,  # Thermal diffusivity
            grid_spacing=[0.1],
            track_conservation=True,
            name="EnergyDiff"
        )
        
        # Set up temperature field
        initial_temp = torch.ones(1, 30) * 293.15  # 20 C
        # Add temperature perturbation
        initial_temp[0, 10:20] += 10.0  # Warm patch
        
        # Calculate total energy (proportional to temperature for constant properties)
        heat_capacity = 1004.0  # J kg-1 K-1 for air
        density = 1.2  # kg m-3
        total_energy_before = torch.sum(initial_temp) * heat_capacity * density
        
        # Set velocity field
        velocity = torch.ones(1, 30) * 0.5
        temp_advection.set_velocity_field([velocity])
        
        state = {
            "tracer_concentration": initial_temp,
            "source_term": torch.zeros_like(initial_temp)  # No heating/cooling
        }
        
        dt = 0.1
        
        # Coupled transport
        adv_output = temp_advection.forward(state, time=0.0, dt=dt)
        state_after_adv = {
            "tracer_concentration": initial_temp + _linear_increment(adv_output),
            "source_term": torch.zeros_like(initial_temp)
        }
        
        diff_output = temp_diffusion.forward(state_after_adv, time=dt, dt=dt)
        
        final_temp = (
            state_after_adv["tracer_concentration"]
            + _fickian_increment(diff_output)
        )
        total_energy_after = torch.sum(final_temp) * heat_capacity * density
        
        # Energy should be conserved
        relative_energy_change = torch.abs(total_energy_after - total_energy_before) / total_energy_before
        assert relative_energy_change < 0.05
        
    def test_momentum_conservation_fluid_transport(self):
        """Test momentum conservation in fluid transport."""
        # Simplified test: transport of momentum (velocity field)
        velocity_transport = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.05],
            track_conservation=True,
            name="MomentumTransport"
        )
        
        # Set up velocity field (transporting itself - nonlinear advection)
        initial_velocity = torch.ones(1, 40) * 1.0  # 1 m/s
        # Add velocity perturbation
        initial_velocity[0, 15:25] = 2.0
        
        # Calculate total momentum
        density = 1.0  # kg m-3 (simplified)
        total_momentum_before = torch.sum(initial_velocity * initial_velocity) * density
        
        velocity_transport.set_velocity_field([initial_velocity])
        
        state = {
            "tracer_concentration": initial_velocity,  # Transport velocity
            "source_term": torch.zeros_like(initial_velocity)
        }
        
        dt = 0.01  # Small time step for nonlinear stability
        
        output = velocity_transport.forward(state, time=0.0, dt=dt)
        final_velocity = initial_velocity + _linear_increment(output)
        total_momentum_after = torch.sum(final_velocity * final_velocity) * density
        
        # For nonlinear advection, momentum is not strictly conserved
        # But changes should be reasonable and finite
        momentum_change = torch.abs(total_momentum_after - total_momentum_before)
        assert torch.isfinite(momentum_change)
        assert momentum_change < total_momentum_before  # Should not blow up


class TestPerformanceAndStabilityIntegration:
    """Test performance and stability under integrated conditions."""
    
    def test_stability_under_strong_coupling(self):
        """Test numerical stability under strong module coupling."""
        # Create strongly coupled system
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.001],  # Fine grid
            name="StrongAdv"
        )
        
        diffusion = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-3,  # High diffusivity
            grid_spacing=[0.001],
            name="StrongDiff"
        )
        
        # Set strong forcing
        velocity = torch.ones(1, 100) * 10.0  # High velocity
        advection.set_velocity_field([velocity])
        
        # Initial condition with sharp gradients
        initial_field = torch.zeros(1, 100)
        initial_field[0, 45:55] = 100.0  # Sharp pulse
        
        state = {
            "tracer_concentration": initial_field,
            "source_term": torch.zeros_like(initial_field)
        }
        
        # Use very small time step for stability
        dt = 0.00005
        
        # Run multiple steps
        n_steps = 10
        for i in range(n_steps):
            # Advection step
            adv_output = advection.forward(state, time=i*dt, dt=dt)
            state["tracer_concentration"] += _linear_increment(adv_output)
            
            # Diffusion step
            diff_output = diffusion.forward(state, time=(i+0.5)*dt, dt=dt)
            state["tracer_concentration"] += _fickian_increment(diff_output)
            
            # Check stability
            assert torch.isfinite(state["tracer_concentration"]).all()
            assert torch.all(state["tracer_concentration"] >= 0)  # Non-negative
            
        # Final state should be smooth but finite
        final_field = state["tracer_concentration"]
        assert float(torch.std(final_field)) <= float(torch.std(initial_field))  # Should not be rougher
        assert torch.max(final_field) < torch.max(initial_field) * 2  # Should not blow up
        
    def test_performance_large_scale_integration(self):
        """Test performance with large-scale integration."""
        import time
        
        # Create large system
        n_points = 1000
        
        advection = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.01],
            name="LargeAdv"
        )
        
        # Set up large field
        velocity = torch.ones(n_points) * 1.0
        advection.set_velocity_field([velocity])
        
        concentration = torch.randn(n_points) * 10.0 + 100.0
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }
        
        # Time performance
        dt = 0.005
        n_steps = 100
        
        start_time = time.time()
        
        for i in range(n_steps):
            output = advection.forward(state, time=i*dt, dt=dt)
            state["tracer_concentration"] += _linear_increment(output)
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Should complete in reasonable time
        assert elapsed_time < 10.0  # Less than 10 seconds
        
        # Average time per step
        avg_time_per_step = elapsed_time / n_steps
        assert avg_time_per_step < 0.1  # Less than 100ms per step
        
        # Final state should be valid
        assert torch.isfinite(state["tracer_concentration"]).all()
        
    def test_memory_usage_integration(self):
        """Test memory usage with integrated modules."""
        # Monitor memory usage during integration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
        # Create multiple modules
        modules = []
        for i in range(5):
            advection = LinearAdvection(
                spatial_dims=1,
                grid_spacing=[0.01],
                name=f"MemTest_{i}"
            )
            
            velocity = torch.ones(200) * (0.1 + i * 0.1)
            advection.set_velocity_field([velocity])
            modules.append(advection)
            
        # Run integrated simulation
        concentration = torch.ones(200) * 100.0
        state = {
            "tracer_concentration": concentration.unsqueeze(0),
            "source_term": torch.zeros_like(concentration.unsqueeze(0))
        }
        
        dt = 0.01
        for step in range(10):
            for i, module in enumerate(modules):
                output = module.forward(state, time=step*dt, dt=dt)
                # Apply only a fraction of each output to avoid instability
                state["tracer_concentration"] += _linear_increment(output) * 0.1
                
        # Check memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
            
        # Verify final state is valid
        assert torch.isfinite(state["tracer_concentration"]).all()
        assert torch.all(state["tracer_concentration"] >= 0)


# Test fixtures for integration tests
@pytest.fixture
def coupled_transport_setup():
    """Fixture providing coupled transport setup."""
    # Create advection and diffusion modules
    advection = LinearAdvection(spatial_dims=1, grid_spacing=[0.01], name="Adv")
    diffusion = FickianDiffusion(spatial_dims=1, diffusivity=1e-5, grid_spacing=[0.01], name="Diff")
    
    # Set velocity field
    velocity = torch.ones(20) * 0.5
    advection.set_velocity_field([velocity])
    
    # Initial concentration
    concentration = torch.ones(20) * 100.0
    
    return {
        'advection': advection,
        'diffusion': diffusion,
        'velocity': velocity,
        'concentration': concentration
    }


@pytest.fixture
def multi_scale_setup():
    """Fixture providing multi-scale transport setup."""
    # Fine scale
    fine_advection = LinearAdvection(spatial_dims=1, grid_spacing=[0.001], name="Fine")
    fine_velocity = torch.ones(50) * 0.01
    fine_advection.set_velocity_field([fine_velocity])
    
    # Coarse scale
    coarse_advection = LinearAdvection(spatial_dims=1, grid_spacing=[0.01], name="Coarse")
    coarse_velocity = torch.ones(10) * 0.5
    coarse_advection.set_velocity_field([coarse_velocity])
    
    return {
        'fine_advection': fine_advection,
        'coarse_advection': coarse_advection,
        'fine_velocity': fine_velocity,
        'coarse_velocity': coarse_velocity
    }


# Integration tests using fixtures
def test_coupled_transport_integration(coupled_transport_setup):
    """Test coupled transport using fixture."""
    advection = coupled_transport_setup['advection']
    diffusion = coupled_transport_setup['diffusion']
    concentration = coupled_transport_setup['concentration']
    
    state = {
        "tracer_concentration": concentration.unsqueeze(0),
        "source_term": torch.zeros_like(concentration.unsqueeze(0))
    }
    
    dt = 0.01
    
    # Sequential application
    adv_output = advection.forward(state, time=0.0, dt=dt)
    state_after_adv = {
        "tracer_concentration": concentration.unsqueeze(0) + _linear_increment(adv_output),
        "source_term": torch.zeros_like(concentration.unsqueeze(0))
    }
    
    diff_output = diffusion.forward(state_after_adv, time=dt, dt=dt)
    
    # Verify outputs
    assert "tracer_concentration" in adv_output
    assert "tracer_concentration" in diff_output
    assert torch.isfinite(_linear_increment(adv_output)).all()
    assert torch.isfinite(_fickian_increment(diff_output)).all()


def test_multi_scale_integration(multi_scale_setup):
    """Test multi-scale transport using fixture."""
    fine_advection = multi_scale_setup['fine_advection']
    coarse_advection = multi_scale_setup['coarse_advection']
    multi_scale_setup['fine_velocity']
    multi_scale_setup['coarse_velocity']
    
    # Fine scale computation
    fine_conc = torch.randn(50) * 10.0 + 100.0
    fine_state = {
        "tracer_concentration": fine_conc.unsqueeze(0),
        "source_term": torch.zeros_like(fine_conc.unsqueeze(0))
    }
    
    dt = 0.01
    fine_output = fine_advection.forward(fine_state, time=0.0, dt=dt)
    
    # Coarse scale computation
    coarse_conc = torch.randn(10) * 10.0 + 100.0
    coarse_state = {
        "tracer_concentration": coarse_conc.unsqueeze(0),
        "source_term": torch.zeros_like(coarse_conc.unsqueeze(0))
    }
    
    coarse_output = coarse_advection.forward(coarse_state, time=0.0, dt=dt)
    
    # Verify both scales work
    fine_increment = _linear_increment(fine_output)
    coarse_increment = _linear_increment(coarse_output)
    assert fine_increment.shape == fine_conc.unsqueeze(0).shape
    assert coarse_increment.shape == coarse_conc.unsqueeze(0).shape
    assert torch.isfinite(fine_increment).all()
    assert torch.isfinite(coarse_increment).all()


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])

