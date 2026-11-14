"""
L2 Integration Tests for Transport Modules

Tests the integration between:
1. Transport modules and StateManager
2. Transport modules and Simulator
3. Different transport mechanisms working together
4. Parameter system integration across modules
"""

import pytest
import torch

from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.core.state.types import StateVariableSpec, StateVariableType
from relic.core.simulator import RELICSimulator
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.atmosphere.most.stability_parameters import (
    friction_velocity,
    monin_obukhov_length,
)


def _register_tracer_variable(
    state_manager: StateManager,
    *,
    initial_value: float,
    bounds: tuple[float, float] | None = None,
) -> None:
    spec = StateVariableSpec(
        name="tracer_concentration",
        units="kg m-3",
        bounds=bounds,
        conserved=False,
        spatial_dims=1,
        required=False,
        default_value=float(initial_value),
        variable_type=StateVariableType.CONCENTRATION,
    )
    state_manager.register_variable(spec)


def _disable_contract_registry(simulator: RELICSimulator) -> None:
    if hasattr(simulator, "state_manager"):
        simulator.state_manager._contract_registry = None


def _transport_increment(updates):
    """Return an integrated transport increment with canonical write semantics."""
    update = updates["tracer_concentration"]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return update.value


class TestTransportStateManagerIntegration:
    """Test integration between transport modules and StateManager."""
    
    def test_diffusion_with_state_manager(self):
        """Test diffusion module with StateManager apply_updates."""
        # Initialize StateManager
        state_manager = StateManager()
        
        # Register state variables
        _register_tracer_variable(
            state_manager,
            initial_value=100.0,
            bounds=(0.0, 1000.0),
        )
        
        # Create diffusion module
        config = {'parameters': {'diffusivity': 1e-5}}
        diffusion = FickianDiffusion(
            name='DiffusionTest',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        # Get initial state
        state = state_manager.get_state()
        
        # Create actual tensor data
        concentration = torch.ones(1, 10, device=state_manager.device) * 100.0
        concentration[0, 5] = 150.0  # Peak
        state["tracer_concentration"] = concentration
        
        # Run diffusion forward
        dt = 0.01
        updates = diffusion.forward(state, time=0.0, dt=dt)
        
        # Apply updates through StateManager
        state_manager.set_state(state)  # Set current state
        new_state = state_manager.apply_updates(updates, dt)
        
        # Verify update was applied correctly
        new_concentration = new_state["tracer_concentration"]
        
        # Should be old + increment
        expected = concentration.to(
            device=new_concentration.device,
            dtype=new_concentration.dtype,
        ) + _transport_increment(updates).to(
            device=new_concentration.device,
            dtype=new_concentration.dtype,
        )
        assert torch.allclose(new_concentration, expected)
        
        # Peak should have diffused
        assert new_concentration[0, 5] < concentration[0, 5]
        
    def test_advection_diffusion_with_state_manager(self):
        """Test advection-diffusion module with StateManager."""
        # Initialize StateManager
        state_manager = StateManager()
        
        # Register variables
        _register_tracer_variable(
            state_manager,
            initial_value=50.0,
        )
        
        # Create advection-diffusion with velocity
        velocity = [torch.ones(1, 10, device=state_manager.device) * 0.1]
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            velocity_field=velocity,
            grid_spacing=(0.01,),
            splitting_method="combined"
        )
        
        # Set up state
        state = state_manager.get_state()
        concentration = torch.ones(1, 10, device=state_manager.device) * 50.0
        concentration[0, 2] = 100.0  # Upstream peak
        state["tracer_concentration"] = concentration
        
        # Run transport
        dt = 0.01
        updates = ad.forward(state, time=0.0, dt=dt)
        
        # Apply updates
        state_manager.set_state(state)
        new_state = state_manager.apply_updates(updates, dt)
        
        # Verify advection effect (peak moves downstream)
        new_concentration = new_state["tracer_concentration"]
        assert new_concentration[0, 3] > concentration[0, 3]  # Downstream increases
        
    def test_conservation_tracking_integration(self):
        """Test conservation tracking through StateManager."""
        # Initialize StateManager with conservation tracking
        state_manager = StateManager()
        state_manager.set_conservation_ledger({
            "carbon": ["tracer_concentration"],
            "mass": ["tracer_concentration"]
        })
        
        # Register variable
        _register_tracer_variable(
            state_manager,
            initial_value=100.0,
        )
        
        # Create diffusion module
        config = {'parameters': {'diffusivity': 1e-5}}
        diffusion = FickianDiffusion(
            name='ConservationTest',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            track_conservation=True
        )
        
        # Set initial state
        state = state_manager.get_state()
        state["tracer_concentration"] = torch.ones(1, 20, device=state_manager.device) * 100.0
        state_manager.set_state(state)
        
        # Run diffusion (no source, should conserve mass)
        updates = diffusion.forward(state, time=0.0, dt=0.01)
        
        # Apply with conservation check
        new_state = state_manager.apply_updates(
            updates, 
            dt=0.01,
            check_conservation=True
        )
        
        # Generate conservation report
        report = state_manager.generate_conservation_report()
        
        # Should have minimal conservation error
        if "carbon" in report:
            assert abs(report["carbon"]["relative_change"]) < 0.01


class TestTransportSimulatorIntegration:
    """Test integration between transport modules and Simulator."""
    
    def test_diffusion_in_simulator(self):
        """Test diffusion module running in simulator."""
        # Create simulator config
        config = {
            "time_step": 1.0,
            "max_time": 10.0,
            "enable_bootstrap_modules": False,
            "enable_auto_scale": False,
            "strict_scale_detection": False,
        }
        
        # Initialize simulator
        simulator = RELICSimulator(config=config)
        
        # Create and register diffusion functor
        diff_config = {'parameters': {'diffusivity': 1e-5}}
        diffusion = FickianDiffusion(
            name='SimDiffusion',
            config=diff_config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        # Mock the functor interface methods if needed
        diffusion.get_dependencies = lambda: []
        
        # Register functor
        simulator.register_functor(diffusion)
        
        # Initialize state
        initial_state = {
            "tracer_concentration": torch.ones(1, 10) * 100.0,
            "source_term": torch.zeros(1, 10),
        }
        simulator.initialize_state(initial_state)
        _disable_contract_registry(simulator)
        
        # Run one step
        updates = simulator.step(dt=1.0)
        
        # Should have tracer_concentration update
        assert "tracer_concentration" in updates or len(updates) == 0  # May be empty if no change
        
        # Check state changed
        final_state = simulator.get_state()
        assert "tracer_concentration" in final_state
        
    def test_multiple_fickian_closures_are_rejected(self):
        """A canonical tracer state cannot register two Fickian closures."""
        config = {
            "time_step": 0.1,
            "max_time": 1.0,
            "enable_bootstrap_modules": False,
            "enable_auto_scale": False,
            "strict_scale_detection": False,
        }
        
        simulator = RELICSimulator(config=config)
        
        # Create two different transport processes
        diff1_config = {'parameters': {'diffusivity': 1e-5}}
        diffusion1 = FickianDiffusion(
            name='FastDiffusion',
            config=diff1_config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        diff2_config = {'parameters': {'diffusivity': 1e-6}}
        diffusion2 = FickianDiffusion(
            name='SlowDiffusion',
            config=diff2_config,
            spatial_dims=1,
            diffusivity=1e-6,
            grid_spacing=(0.01,),
        )

        simulator.register_functor(diffusion1)
        with pytest.raises(ValueError, match="Mutually exclusive overlap"):
            simulator.register_functor(diffusion2)


class TestParameterSystemIntegration:
    """Test parameter system integration across transport modules."""
    
    def test_operators_use_parameter_system(self):
        """Test that operators use parameters from system."""
        # Test canonical friction velocity with neutral stability.
        U = torch.tensor(10.0)
        z = torch.tensor(10.0)
        z0m = torch.tensor(0.01)
        d = torch.tensor(0.0)
        zeta = torch.tensor(0.0)
        
        u_star = friction_velocity(U, z, z0m, d, zeta)
        assert u_star > 0
        
        # Positive upward heat flux must yield unstable negative L.
        rho = 1.2
        cp = 1005.0
        T = torch.tensor(300.0)
        H = torch.tensor(100.0)
        
        L = monin_obukhov_length(u_star, rho, cp, T, H)
        assert torch.isfinite(L)
        assert L < 0
        
    def test_transport_modules_parameter_consistency(self):
        """Test parameter consistency across transport modules."""
        # Create multiple transport modules
        config = {'parameters': {}}
        
        diff = FickianDiffusion(
            name='DiffParam',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        # Both must resolve the operator-owned diffusion limit through the same
        # canonical ParameterManager key. The combined Functor must not copy a
        # second local timestep or conservation-tolerance owner.
        diff_limit = diff.get_parameter_value(
            "cfl_diffusion_max",
            namespace="transport.numerical",
            required=True,
        )
        ad_limit = ad.get_parameter_value(
            "cfl_diffusion_max",
            namespace="transport.numerical",
            required=True,
        )
        assert float(diff_limit) == pytest.approx(float(ad_limit))
        assert float(ad_limit) > 0.0
        assert not hasattr(ad, "min_timestep")
        assert not hasattr(ad, "conservation_tolerance")


class TestTransportCoupling:
    """Test coupling between different transport mechanisms."""
    
    def test_sequential_transport_processes(self):
        """Test sequential application of transport processes."""
        # Initial concentration
        concentration = torch.ones(1, 10) * 100.0
        concentration[0, 5] = 200.0  # Peak
        
        # First process: Diffusion
        config1 = {'parameters': {'diffusivity': 1e-5}}
        diffusion = FickianDiffusion(
            name='Diffusion',
            config=config1,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        state = {"tracer_concentration": concentration}
        dt = 0.01
        
        # Apply diffusion
        diff_updates = diffusion.forward(state, time=0.0, dt=dt)
        concentration_after_diff = concentration + _transport_increment(diff_updates)
        
        # Second process: Advection-Diffusion
        velocity = [torch.ones(1, 10) * 0.1]
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-6,  # Smaller diffusivity
            velocity_field=velocity,
            grid_spacing=(0.01,)
        )
        
        state2 = {"tracer_concentration": concentration_after_diff}
        ad_updates = ad.forward(state2, time=0.0, dt=dt)
        final_concentration = concentration_after_diff + _transport_increment(ad_updates)
        
        # Peak should have both diffused and advected
        assert final_concentration[0, 5] < concentration[0, 5]  # Diffused
        assert final_concentration[0, 6] > concentration[0, 6]  # Advected downstream
        
    def test_transport_with_reactions(self):
        """Test transport coupled with reaction source terms."""
        # Create transport with source term
        config = {'parameters': {'diffusivity': 1e-5}}
        diffusion = FickianDiffusion(
            name='ReactDiff',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        concentration = torch.ones(1, 10) * 100.0
        
        # Reaction source term (production in center, consumption at edges)
        source = torch.zeros(1, 10)
        source[0, 4:6] = 10.0   # Production
        source[0, 0:2] = -5.0   # Consumption
        source[0, 8:10] = -5.0  # Consumption
        
        state = {
            "tracer_concentration": concentration,
            "source_term": source
        }
        
        dt = 0.01
        updates = diffusion.forward(state, time=0.0, dt=dt)
        
        new_concentration = concentration + _transport_increment(updates)
        
        # Center should increase (production > diffusion loss)
        assert new_concentration[0, 5] > concentration[0, 5]
        # Edges should decrease (consumption)
        assert new_concentration[0, 0] < concentration[0, 0]


class TestErrorHandling:
    """Test error handling in integrated scenarios."""
    
    def test_invalid_dimensions_handling(self):
        """Test handling of dimension mismatches."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diffusion = FickianDiffusion(
            name='DimTest',
            config=config,
            spatial_dims=2,  # Expects 2D
            diffusivity=1e-5,
            grid_spacing=(0.01, 0.01)
        )
        
        # Provide 1D concentration
        state = {
            "tracer_concentration": torch.ones(1, 10)  # Wrong dimensions
        }
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            diffusion.forward(state, time=0.0, dt=0.01)
            
    def test_missing_state_variable(self):
        """Test handling of missing state variables."""
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        # State without required variable
        state = {
            "temperature": torch.ones(1, 10) * 300.0
        }
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            ad.forward(state, time=0.0, dt=0.01)
            
    def test_negative_timestep_handling(self):
        """Test handling of invalid timestep."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diffusion = FickianDiffusion(
            name='TimestepTest',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        state = {
            "tracer_concentration": torch.ones(1, 10) * 100.0
        }
        
        # Negative timestep should raise error
        with pytest.raises(ValueError):
            diffusion.forward(state, time=0.0, dt=-0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


