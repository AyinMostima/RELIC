"""
L2 integration test to verify transport module update semantics.

This test verifies that most transport functors return increments (deltas), while
solute transport returns absolute concentrations with UpdateMode.REPLACE.
"""

import pytest
import torch

# Import transport modules
from relic.scientific.transport.diffusion import FickianDiffusion
from relic.scientific.transport.advection import LinearAdvection
from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
from relic.scientific.pedosphere.transport.solute_advection_diffusion import SoluteAdvectionDiffusion1D

# Import supporting classes
from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


def _transport_increment(output):
    """Return an integrated transport update with canonical write semantics."""
    update = output["tracer_concentration"]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    return update.value


class TestIncrementSemantics:
    """Test suite to verify transport module update semantics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.tolerance = 1e-6
        
    def test_diffusion_returns_increments(self):
        """Verify FickianDiffusion returns increments, not new state."""
        # Create diffusion module
        diff = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=[0.01],
            device=self.device,
            name="TestDiff"
        )
        
        # Initial concentration: step profile to ensure non-zero Laplacian
        initial = torch.zeros(1, 10)
        initial[0, :5] = 100.0  # 100 kg m-3 in left half
        initial[0, 5:] = 50.0   # 50 kg m-3 in right half
        state = {"tracer_concentration": initial}
        
        # Forward pass
        output = diff.forward(state, time=0.0, dt=0.1)
        
        # Verify we get an increment
        assert "tracer_concentration" in output
        delta = _transport_increment(output)
        
        # Increment should be small compared to initial value
        assert torch.abs(delta).max() < 10.0  # Should not be ~100
        
        # Apply increment to get new state
        new_concentration = initial + delta
        
        # New concentration should be different from initial
        assert not torch.allclose(new_concentration, initial, atol=1e-8)
        
        # But should be close (small timestep)
        assert torch.allclose(new_concentration, initial, rtol=0.1)
        
    def test_advection_returns_increments(self):
        """Verify LinearAdvection returns increments, not new state."""
        # Create advection module
        adv = LinearAdvection(
            spatial_dims=1,
            grid_spacing=[0.01],
            device=self.device,
            name="TestAdv",
            track_conservation=False
        )
        
        # Set velocity field
        velocity = torch.ones(1, 10) * 0.1  # 0.1 m/s
        adv.set_velocity_field([velocity])
        
        # Initial concentration - step function
        initial = torch.zeros(1, 10)
        initial[0, :5] = 100.0  # 100 kg m-3 in left half
        state = {"tracer_concentration": initial}
        
        # Forward pass
        output = adv.forward(state, time=0.0, dt=0.01)
        
        # Verify we get an increment
        assert "tracer_concentration" in output
        delta = _transport_increment(output)
        
        # Increment should represent the change
        assert torch.abs(delta).max() > 0  # Should have some change
        assert torch.abs(delta).max() < 100.0  # Should not be the full value
        
        # Apply increment
        new_concentration = initial + delta
        
        # Verify physical behavior (advection should move the step)
        assert new_concentration[0, 5] > initial[0, 5]  # Interface should blur
        
    def test_advection_diffusion_returns_increments(self):
        """Verify AdvectionDiffusion returns increments."""
        # Create combined transport
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=[0.01],
            velocity_field=torch.ones(1, 10) * 0.05,
            device=self.device,
            name="TestAD",
            splitting_method="sequential"
        )
        
        # Initial Gaussian pulse
        x = torch.linspace(0, 1, 10)
        initial = torch.exp(-((x - 0.3) ** 2) / (2 * 0.05**2)).unsqueeze(0) * 100.0
        state = {"tracer_concentration": initial}
        
        # Forward pass
        output = ad.forward(state, time=0.0, dt=0.01)
        
        # Verify increment
        assert "tracer_concentration" in output
        delta = _transport_increment(output)
        
        # Should be an increment
        assert torch.abs(delta).max() < torch.abs(initial).max()
        
        # Apply and verify conservation (approximately)
        new_concentration = initial + delta
        assert new_concentration.min() >= -self.tolerance  # No negative
        
    def test_solute_transport_returns_absolute(self):
        """Verify SoluteAdvectionDiffusion1D returns absolute concentrations."""
        # Create solute transport
        n_layers = 5
        solute = SoluteAdvectionDiffusion1D(
            n_layers=n_layers,
            soil_depth=n_layers * 0.1,
            layer_thickness_ratio=1.0,
        )
        
        # Initial state
        initial_no3 = torch.ones(n_layers) * 10.0  # 10 g N m-3
        initial_doc = torch.ones(n_layers) * 5.0   # 5 g C m-3
        
        state = {
            "no3_concentration": initial_no3,
            "doc_concentration": initial_doc,
            "soil_water_content": torch.ones(n_layers) * 0.3,
            "darcy_flux": torch.ones(n_layers) * 1e-6,  # Small flux
            "bulk_density": torch.ones(n_layers) * 1300.0,
            "porosity": torch.ones(n_layers) * 0.45,
            "soil_temperature": torch.ones(n_layers) * 288.15,
            "surface_no3_concentration": torch.tensor(10.0),
            "surface_doc_concentration": torch.tensor(5.0),
        }
        
        # Forward pass
        output = solute.forward(state, time=0.0, dt=60.0)  # 1 minute
        
        # Check NO3 update
        if "no3_concentration" in output:
            no3_update = output["no3_concentration"]
            assert isinstance(no3_update, UpdateItem)
            assert no3_update.mode == UpdateMode.REPLACE
            assert no3_update.per_timestep is True
            new_no3 = no3_update.value
            # Should be close to original for small flux
            assert torch.allclose(new_no3, initial_no3, rtol=0.1)
            
        # Check DOC update
        if "doc_concentration" in output:
            doc_update = output["doc_concentration"]
            assert isinstance(doc_update, UpdateItem)
            assert doc_update.mode == UpdateMode.REPLACE
            assert doc_update.per_timestep is True
            new_doc = doc_update.value
            # Should be close to original for small flux
            assert torch.allclose(new_doc, initial_doc, rtol=0.1)
            
    def test_state_manager_integration(self):
        """Test that StateManager correctly applies transport increments."""
        # Create state manager
        state_mgr = StateManager(config={}, device=self.device)
        
        # Register variables
        state_mgr.register_variable(
            StateVariableSpec(
                name="tracer_concentration",
                units="kg m-3",
                bounds=(0.0, 1000.0),
                default_value=100.0,
            )
        )
        
        # Create simple diffusion
        diff = FickianDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=[0.01],
            device=self.device,
            name="TestStateIntegration"
        )
        
        # Initial state: step profile to ensure non-zero diffusion increment
        initial = torch.zeros(1, 10, device=state_mgr.device)
        initial[0, :5] = 100.0
        initial[0, 5:] = 50.0
        state_mgr.set_state("tracer_concentration", initial)
        
        # Get transport increment
        state = state_mgr.get_state()
        output = diff.forward(state, time=0.0, dt=0.1)
        
        # Apply increment through state manager
        state_mgr.apply_updates(output, dt=0.1)
        
        # Get new state
        new_state = state_mgr.get_state()
        new_concentration = new_state["tracer_concentration"]
        
        # Verify increment was applied correctly
        expected = initial + _transport_increment(output)
        assert torch.allclose(new_concentration, expected, atol=1e-10)
        
        # Concentration should have changed slightly
        assert not torch.allclose(new_concentration, initial, atol=1e-10)
        

class TestParameterIntegration:
    """Test parameter system integration in transport modules."""
    
    def test_operators_use_parameter_system(self):
        """Verify canonical MOST operators use physical-constant SSOT."""
        from relic.scientific.atmosphere.most.stability_parameters import (
            friction_velocity,
            monin_obukhov_length,
        )
        
        # Test u_star computation (uses von Karman constant)
        U = torch.tensor(10.0)  # Wind speed
        z = torch.tensor(10.0)  # Height
        d = torch.tensor(0.0)   # Displacement
        z0m = torch.tensor(0.01) # Roughness
        zeta = torch.tensor(0.0)  # Neutral stability parameter
        
        u_star = friction_velocity(U, z, z0m, d, zeta)
        assert u_star > 0
        
        # Test L computation (uses gravity)
        rho = 1.225  # Air density
        cp = 1004.0  # Specific heat
        T = torch.tensor(288.15)  # Temperature
        H = torch.tensor(100.0)   # Heat flux
        
        L = monin_obukhov_length(u_star, rho, cp, T, H)
        assert L < 0
        
    def test_transport_modules_parameter_access(self):
        """Verify transport modules can access numerical parameters."""
        from relic.scientific.transport.advection_diffusion import AdvectionDiffusion
        
        # Create module with default parameters
        ad = AdvectionDiffusion(
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=[0.01],
            device=torch.device("cpu"),
            name="TestParams"
        )
        
        expected_parameter_keys = {
            "transport.advection.default_velocity_magnitude",
            "transport.diffusion.diffusivity",
            "transport.advection_diffusion.grid_spacing",
            "transport.advection_diffusion.splitting_method",
            "transport.numerical.cfl_advection_max",
            "transport.numerical.cfl_diffusion_max",
            "transport.numerical.default_boundary_type",
            "transport.numerical.enforce_non_negative",
            "transport.numerical.track_conservation",
        }
        assert set(ad.required_parameters()) == expected_parameter_keys
        assert set(ad.metadata.parameters) == expected_parameter_keys
        assert not hasattr(ad, "safety_factor")
        assert not hasattr(ad, "auto_adjust_timestep")
        assert not hasattr(ad, "min_timestep")
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

