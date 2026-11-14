"""
L1 Unit Tests for Diffusion Module

Tests the diffusion module to ensure:
1. Returns increments instead of new states
2. Uses parameter system for stability parameters
3. Different integration schemes work correctly
4. Anisotropic diffusion is handled properly
"""

import pytest
import torch
from types import SimpleNamespace
import relic.core.functor.functor_base as functor_base_module
from relic.parameters import UnifiedParameterManager, ParameterSpec
from relic.core.simulator import RELICSimulator

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.scientific.transport.diffusion import (
    FickianDiffusion,
    create_isotropic_diffusion,
    create_anisotropic_diffusion
)
from relic.core.state import UpdateItem, UpdateMode
from relic.orchestration.node_registry import NodeRegistry
from relic.orchestration.ssot_catalog import SSOTCatalog


def _tracer_increment(output):
    """Return the integrated tracer increment after validating write semantics."""
    update = output["tracer_concentration"]
    assert isinstance(update, UpdateItem)
    assert update.mode == UpdateMode.ACCUMULATE
    assert update.per_timestep is True
    assert isinstance(update.value, torch.Tensor)
    return update.value


class TestIncrementReturns:
    """Test that the module returns increments, not new states."""
    
    def test_forward_returns_increments_1d(self):
        """Test that forward returns concentration increments in 1D."""
        # Create 1D diffusion
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestDiffusion',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            time_integration="explicit"
        )
        
        # Create initial state with a peak
        concentration = torch.ones(1, 20) * 50.0
        concentration[0, 10] = 100.0  # Peak in center
        
        state = {
            "tracer_concentration": concentration,
            "source_term": torch.zeros_like(concentration)
        }
        
        # Run forward step
        dt = 0.01
        output = diff.forward(state, time=0.0, dt=dt)
        
        # Check that output contains increments
        assert "tracer_concentration" in output
        delta = _tracer_increment(output)
        
        # Peak should diffuse: negative at center, positive around
        assert delta[0, 10] < 0  # Center decreases
        assert delta[0, 9] > 0   # Neighbors increase
        assert delta[0, 11] > 0

        # Total change should be small
        assert torch.all(torch.abs(delta) < 20.0)
        torch.testing.assert_close(
            torch.sum(delta),
            torch.zeros((), dtype=delta.dtype),
            rtol=0.0,
            atol=1.0e-6,
        )
        
    def test_forward_returns_increments_2d(self):
        """Test that forward returns concentration increments in 2D."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestDiffusion2D',
            config=config,
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=(0.01, 0.01),
            time_integration="explicit"
        )
        
        # Create 2D Gaussian concentration
        x = torch.linspace(-1, 1, 10)
        y = torch.linspace(-1, 1, 10)
        Y, X = torch.meshgrid(y, x, indexing='ij')
        concentration = torch.exp(-(X**2 + Y**2) / 0.2).unsqueeze(0) * 100
        
        state = {
            "tracer_concentration": concentration
        }
        
        output = diff.forward(state, time=0.0, dt=0.001)
        delta = _tracer_increment(output)
        
        # Center should decrease (negative increment)
        center_idx = 5
        assert delta[0, center_idx, center_idx] < 0
        
        # Edges should increase slightly
        assert delta[0, 2, center_idx] > -1.0  # Less negative or positive
        torch.testing.assert_close(
            torch.sum(delta),
            torch.zeros((), dtype=delta.dtype),
            rtol=0.0,
            atol=1.0e-5,
        )
        
    def test_forward_returns_increments_3d(self):
        """Test that forward returns concentration increments in 3D."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestDiffusion3D',
            config=config,
            spatial_dims=3,
            diffusivity=1e-5,
            grid_spacing=(0.01, 0.01, 0.01),
            time_integration="explicit"
        )
        
        # Create 3D concentration with central peak
        concentration = torch.ones(1, 5, 5, 5) * 10.0
        concentration[0, 2, 2, 2] = 50.0  # Central peak
        
        state = {"tracer_concentration": concentration}
        
        output = diff.forward(state, time=0.0, dt=0.0001)
        delta = _tracer_increment(output)
        
        # Central peak should decrease
        assert delta[0, 2, 2, 2] < 0
        
        # Neighbors should increase
        assert delta[0, 2, 2, 1] > 0  # x-neighbor
        assert delta[0, 2, 1, 2] > 0  # y-neighbor
        assert delta[0, 1, 2, 2] > 0  # z-neighbor
        torch.testing.assert_close(
            torch.sum(delta),
            torch.zeros((), dtype=delta.dtype),
            rtol=0.0,
            atol=1.0e-5,
        )
        
    def test_diagnostics_have_prefix(self):
        """Test that diagnostic outputs have proper prefix."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestDiagnostics',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            track_conservation=False,
        )
        
        state = {
            "tracer_concentration": torch.ones(1, 10) * 100.0
        }
        
        output = diff.forward(state, time=0.0, dt=0.1)

        assert set(output) == {
            "tracer_concentration",
            "tracer_concentration_diffusive_boundary_flux",
            "diagnostics.diffusion_stability",
        }
        assert "tracer_concentration_diffusive_boundary_flux" in diff.metadata.output_variables
        assert "diagnostics.conservation_error" in diff.metadata.optional_output_variables
        for key in output:
            if key not in {
                "tracer_concentration",
                "tracer_concentration_diffusive_boundary_flux",
            }:
                assert key.startswith("diagnostics.")

    @pytest.mark.parametrize(
        "legacy_key",
        (
            "__updated_tracer_concentration",
            "concentration_field",
            "concentration",
        ),
    )
    def test_legacy_concentration_aliases_are_rejected(self, legacy_key):
        """Test that production forward accepts only the declared tracer key."""
        diff = FickianDiffusion(
            name="TestCanonicalInput",
            spatial_dims=1,
            diffusivity=1.0e-5,
            grid_spacing=(0.01,),
        )

        with pytest.raises(ValueError, match="tracer_concentration"):
            diff.forward({legacy_key: torch.ones(1, 10)}, time=0.0, dt=0.01)

    def test_dynamic_tracer_identity_is_rejected(self):
        with pytest.raises(TypeError, match="tracer_key"):
            FickianDiffusion(
                spatial_dims=1,
                diffusivity=1.0e-5,
                tracer_key="temperature",
            )


class TestParameterSystemIntegration:
    """Test parameter system integration."""

    @pytest.fixture
    def bound_diffusion(self, monkeypatch):
        manager = UnifiedParameterManager(device="cpu")
        key = "transport.diffusion.live_bound_test"
        manager.register_parameter(ParameterSpec(
            key=key, value=0.25, units="m2 s-1", dtype="float64", bounds=(0.0, 1.0),
            requires_grad=True, references=["Crank (1975), The Mathematics of Diffusion"], category="model_parameter",
        ))
        monkeypatch.setattr(functor_base_module, "_get_parameter_manager", lambda: manager)
        functor = FickianDiffusion(
            spatial_dims=1, diffusivity_parameter_key=key, grid_spacing=[1.0],
            boundary_method="zero", non_negative=False, track_conservation=False,
        )
        return manager, key, functor

    def test_default_diffusivity_keeps_the_parameter_gradient(self, bound_diffusion):
        manager, key, functor = bound_diffusion
        state = {functor.tracer_key: torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)}
        increment = _tracer_increment(functor.forward(state, time=0.0, dt=0.125))
        assert increment.requires_grad
        increment[0].backward()
        torch.testing.assert_close(manager.nn_parameters[key].grad, torch.tensor(0.125, dtype=torch.float64))

    def test_parameter_update_changes_production_and_structural_audits(self, bound_diffusion):
        manager, key, functor = bound_diffusion
        state = {functor.tracer_key: torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)}
        initial = _tracer_increment(functor.forward(state, time=0.0, dt=0.125))
        manager.set_value(key, 0.5)
        updated = _tracer_increment(functor.forward(state, time=0.0, dt=0.125))
        torch.testing.assert_close(updated, 2.0 * initial)
        torch.testing.assert_close(functor.entropy_terms(state)["dissipation_rate"], torch.tensor(1.0, dtype=torch.float64))
        torch.testing.assert_close(functor.build_hodge_contract(n_dofs=3).matrix.diag(), torch.full((3,), 0.5, dtype=torch.float64))

    @pytest.mark.parametrize("override", [0.75, torch.tensor(0.75, dtype=torch.float64)])
    def test_explicit_diffusivity_override_stays_explicit(self, bound_diffusion, override):
        manager, key, functor = bound_diffusion
        functor.set_diffusivity(override)
        manager.set_value(key, 0.5)
        assert functor.get_diffusivity().item() == 0.75

    def test_resolved_hodge_candidate_does_not_change_parameter_binding(self, bound_diffusion):
        manager, key, functor = bound_diffusion
        with pytest.raises(ValueError, match="coefficients"):
            functor.build_hodge_terms(
                complex_bundle=SimpleNamespace(n_dofs=3),
                resolved_parameters={key: torch.tensor([0.75, 0.8], dtype=torch.float64)},
            )
        manager.set_value(key, 0.5)
        assert functor.get_diffusivity().item() == 0.5

    def test_live_diffusivity_updates_reach_the_governed_simulator(self, bound_diffusion):
        manager, key, functor = bound_diffusion
        simulator = RELICSimulator({"device": "cpu", "conservation_checks": False, "enable_auto_scale": False})
        simulator.initialize_state({functor.tracer_key: torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)})
        simulator.register_functor(functor, execution_group="fast")
        simulator.step(dt=0.125)
        first = simulator.state_manager.get_current_state()[functor.tracer_key]
        torch.testing.assert_close(first, torch.tensor([0.03125, 0.9375, 0.03125], dtype=torch.float64))
        assert first.requires_grad
        manager.set_value(key, 0.5)
        simulator.step(dt=0.125)
        second = simulator.state_manager.get_current_state()[functor.tracer_key]
        torch.testing.assert_close(second, torch.tensor([0.087890625, 0.82421875, 0.087890625], dtype=torch.float64))
        assert simulator.step_count == 2
        assert simulator.current_time == 0.25
        torch.testing.assert_close(second.sum(), torch.tensor(1.0, dtype=torch.float64))
    
    def test_parameters_loaded_from_system(self):
        """Numerical controls stay owned by the canonical operator and scheduler."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestParams',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )

        assert not hasattr(diff, 'max_cfl_diffusion')
        assert not hasattr(diff, 'min_timestep')
        assert not hasattr(diff, 'conservation_tolerance')
        assert "transport.numerical.cfl_diffusion_max" in diff.required_parameters()
        assert "transport.numerical.track_conservation" in diff.required_parameters()

    def test_configurable_diffusivity_parameter_key_is_single_source(self):
        """A case-specific K must bind runtime, metadata, and Hodge evidence."""
        parameter_key = (
            "transport.diffusion.acp_manufactured_constant_eddy_diffusivity"
        )
        diff = FickianDiffusion(
            name="TestACPManufacturedK",
            spatial_dims=1,
            diffusivity_parameter_key=parameter_key,
            grid_spacing=(10.0,),
        )

        assert diff.diffusivity_scalar == pytest.approx(1.0)
        assert parameter_key in diff.required_parameters()
        assert "transport.diffusion.diffusivity" not in diff.required_parameters()
        assert parameter_key in diff.metadata.parameters
        assert diff.build_hodge_contract().parameter_keys == (parameter_key,)

    def test_configurable_grid_spacing_parameter_key_is_single_source(self):
        """A graph-safe string selector must bind runtime and DEC metric evidence."""
        parameter_key = (
            "transport.diffusion.acp_manufactured_vertical_grid_spacing"
        )
        diff = FickianDiffusion(
            name="TestACPManufacturedGrid",
            spatial_dims=1,
            diffusivity=1.0,
            grid_spacing_parameter_key=parameter_key,
        )

        assert diff.grid_spacing == [pytest.approx(10.0)]
        assert parameter_key in diff.required_parameters()
        assert "transport.diffusion.grid_spacing" not in diff.required_parameters()
        assert parameter_key in diff.metadata.parameters
        assert diff.build_hodge_contract().metric_keys == (parameter_key,)

    def test_registry_requires_only_the_tracer_state(self):
        """Optional source and boundary state ports must remain optional."""
        diff = FickianDiffusion(
            name="TestOptionalGraphInputs",
            spatial_dims=1,
            diffusivity=1.0,
            grid_spacing=(10.0,),
            boundary_method="zero",
        )
        definition = NodeRegistry(SSOTCatalog()).register_functor_instance(
            diff
        )

        assert diff.get_required_inputs() == ["tracer_concentration"]
        assert definition.inputs["tracer_concentration"].required is True
        assert definition.inputs["source_term"].required is False
        assert (
            definition.inputs[
                "tracer_concentration_boundary_state"
            ].required
            is False
        )

    def test_local_timestep_adjustment_alias_is_rejected(self):
        """Only the simulator scheduler may split an unstable requested step."""
        with pytest.raises(TypeError, match="auto_adjust_timestep"):
            FickianDiffusion(
                name='TestStability',
                spatial_dims=1,
                diffusivity=1e-3,
                grid_spacing=(0.001,),
                auto_adjust_timestep=True,
            )

    def test_unstable_explicit_timestep_fails_fast(self):
        """Test that an unstable explicit step cannot run without adjustment."""
        diff = FickianDiffusion(
            name="TestUnstableFailFast",
            spatial_dims=1,
            diffusivity=1.0e-3,
            grid_spacing=(0.001,),
        )
        state = {"tracer_concentration": torch.ones(1, 10)}

        with pytest.raises(ValueError, match="Diffusion stability violation"):
            diff.forward(state, time=0.0, dt=1.0)


class TestIntegrationSchemes:
    """Test different time integration schemes."""
    
    def test_explicit_integration(self):
        """Test explicit Euler integration."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestExplicit',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            time_integration="explicit"
        )
        
        concentration = torch.ones(1, 10) * 50.0
        concentration[0, 5] = 100.0
        state = {"tracer_concentration": concentration}
        
        output = diff.forward(state, time=0.0, dt=0.001)
        delta = _tracer_increment(output)
        
        # Should diffuse the peak
        assert delta[0, 5] < 0
        assert delta[0, 4] > 0
        assert delta[0, 6] > 0
        
    def test_semi_implicit_integration(self):
        """Test semi-implicit integration scheme."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestSemiImplicit',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            time_integration="semi_implicit"
        )
        
        concentration = torch.exp(-((torch.linspace(-1, 1, 10)**2) / 0.1)).unsqueeze(0) * 100
        state = {"tracer_concentration": concentration}
        
        output = diff.forward(state, time=0.0, dt=0.001)
        delta = _tracer_increment(output)
        
        # Should still produce reasonable diffusion
        assert not torch.allclose(delta, torch.zeros_like(delta))
        
        # Semi-implicit should be stable with larger timesteps
        output_large_dt = diff.forward(state, time=0.0, dt=0.01)
        delta_large = _tracer_increment(output_large_dt)
        
        # Should still be bounded
        assert torch.all(torch.abs(delta_large) < 100)


class TestAnisotropicDiffusion:
    """Test anisotropic (directional) diffusion."""
    
    def test_anisotropic_2d(self):
        """Test anisotropic diffusion in 2D."""
        # Different diffusivities in x and y
        diffusivity_tensor = torch.tensor([1e-5, 1e-4])  # Ky, Kx (10x stronger in x)
        
        config = {'parameters': {}}
        diff = FickianDiffusion(
            name='TestAnisotropic2D',
            config=config,
            spatial_dims=2,
            diffusivity=diffusivity_tensor,
            diffusivity_layout="directional",
            grid_spacing=(0.01, 0.01),
            time_integration="explicit"
        )
        
        # Create concentration with central peak
        concentration = torch.ones(1, 10, 10) * 50.0
        concentration[0, 5, 5] = 100.0
        
        state = {"tracer_concentration": concentration}
        output = diff.forward(state, time=0.0, dt=0.0001)
        delta = _tracer_increment(output)
        
        # Should diffuse more in x-direction (higher diffusivity)
        # Check x-neighbors vs y-neighbors
        x_neighbor_delta = abs(delta[0, 5, 6].item())  # x-direction
        y_neighbor_delta = abs(delta[0, 6, 5].item())  # y-direction
        
        # x-diffusion should be stronger (approximately 10x)
        assert x_neighbor_delta > y_neighbor_delta * 5
        
    def test_anisotropic_3d(self):
        """Test anisotropic diffusion in 3D."""
        # Different diffusivities in x, y, z
        diffusivity_tensor = torch.tensor([1e-5, 2e-5, 4e-5])  # Kz, Ky, Kx
        
        config = {'parameters': {}}
        diff = FickianDiffusion(
            name='TestAnisotropic3D',
            config=config,
            spatial_dims=3,
            diffusivity=diffusivity_tensor,
            diffusivity_layout="directional",
            grid_spacing=(0.01, 0.01, 0.01),
            time_integration="explicit"
        )
        
        # Create concentration with central peak
        concentration = torch.ones(1, 5, 5, 5) * 25.0
        concentration[0, 2, 2, 2] = 100.0
        
        state = {"tracer_concentration": concentration}
        output = diff.forward(state, time=0.0, dt=0.00001)
        delta = _tracer_increment(output)
        
        # Check relative diffusion strengths
        assert delta[0, 2, 2, 2] < 0  # Center decreases
        
        # Neighbors should increase proportionally to diffusivity
        z_delta = abs(delta[0, 1, 2, 2].item())  # z-direction (Kz = 1e-5)
        y_delta = abs(delta[0, 2, 1, 2].item())  # y-direction (Ky = 2e-5)
        x_delta = abs(delta[0, 2, 2, 1].item())  # x-direction (Kx = 4e-5)
        
        # Ratios should roughly match diffusivity ratios
        assert x_delta > y_delta  # Kx > Ky
        assert y_delta > z_delta  # Ky > Kz


class TestConservation:
    """Test mass conservation properties."""
    
    def test_conservation_no_source(self):
        """Test mass conservation without source."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestConservation',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            track_conservation=True
        )
        
        concentration = torch.ones(1, 20) * 100.0
        state = {"tracer_concentration": concentration}
        
        output = diff.forward(state, time=0.0, dt=0.001)
        
        # Check conservation error if reported
        if "diagnostics.conservation_error" in output:
            error = output["diagnostics.conservation_error"]
            torch.testing.assert_close(error, torch.zeros_like(error))
            
    def test_conservation_with_source(self):
        """Test mass conservation with source term."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestConservationSource',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        concentration = torch.ones(1, 20) * 100.0
        source = torch.ones(1, 20) * 2.0  # Uniform source
        
        state = {
            "tracer_concentration": concentration,
            "source_term": source
        }
        
        dt = 0.01
        output = diff.forward(state, time=0.0, dt=dt)
        delta = _tracer_increment(output)
        
        # Total mass change should match source integral
        total_delta = torch.sum(delta)
        expected_from_source = torch.sum(source) * dt
        
        # Diffusion redistributes while the open source controls total mass.
        torch.testing.assert_close(
            total_delta,
            expected_from_source,
            rtol=1.0e-5,
            atol=1.0e-6,
        )


class TestBoundaryConditions:
    """Test boundary condition handling."""

    def test_neumann_boundary_emits_explicit_zero_flux_port(self):
        """Closed faces remain visible as an oriented zero cochain."""
        diff = FickianDiffusion(
            spatial_dims=1,
            diffusivity=0.1,
            grid_spacing=(1.0,),
            boundary_method="neumann",
            track_conservation=True,
        )
        concentration = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]],
            dtype=torch.float64,
        )

        output = diff.forward(
            {"tracer_concentration": concentration},
            time=0.0,
            dt=0.1,
        )

        boundary_update = output[
            "tracer_concentration_diffusive_boundary_flux"
        ]
        assert isinstance(boundary_update, UpdateItem)
        assert boundary_update.mode == UpdateMode.REPLACE
        assert boundary_update.per_timestep is False
        assert boundary_update.unit == "kg m-2 s-1"
        torch.testing.assert_close(
            boundary_update.value,
            torch.zeros((1, 2), dtype=torch.float64),
        )
        torch.testing.assert_close(
            output["diagnostics.conservation_error"],
            torch.zeros((), dtype=torch.float64),
        )

    def test_dirichlet_boundary_exposes_outward_flux_and_closes_ledger(self):
        """Dirichlet exchange must leave through the canonical matter port."""
        diff = FickianDiffusion(
            name='TestBC',
            spatial_dims=1,
            diffusivity=0.1,
            grid_spacing=(1.0,),
            boundary_method="dirichlet",
            track_conservation=True,
        )
        concentration = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        boundary_state = torch.zeros_like(concentration)
        output = diff.forward(
            {
                "tracer_concentration": concentration,
                "tracer_concentration_boundary_state": boundary_state,
            },
            time=0.0,
            dt=0.1,
        )
        delta = _tracer_increment(output)
        torch.testing.assert_close(
            delta,
            torch.tensor([[-0.01, 0.0, 0.0, -0.09]], dtype=torch.float64),
        )
        boundary_update = output["tracer_concentration_diffusive_boundary_flux"]
        assert isinstance(boundary_update, UpdateItem)
        assert boundary_update.mode == UpdateMode.REPLACE
        assert boundary_update.per_timestep is False
        assert boundary_update.unit == "kg m-2 s-1"
        torch.testing.assert_close(
            boundary_update.value,
            torch.tensor([[0.2, 0.8]], dtype=torch.float64),
        )
        storage_change = torch.sum(delta)
        outward_export = 0.1 * torch.sum(boundary_update.value)
        torch.testing.assert_close(storage_change + outward_export, torch.zeros(()))
        torch.testing.assert_close(
            output["diagnostics.conservation_error"],
            torch.zeros(()),
        )

    def test_dirichlet_functor_call_uses_routed_mass_ledger(self):
        diff = FickianDiffusion(
            spatial_dims=1,
            diffusivity=0.1,
            grid_spacing=(1.0,),
            boundary_method="dirichlet",
            track_conservation=True,
        )
        concentration = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        output = diff(
            {
                "tracer_concentration": concentration,
                "tracer_concentration_boundary_state": torch.zeros_like(
                    concentration
                ),
            },
            time=0.0,
            dt=0.1,
        )

        torch.testing.assert_close(
            output["diagnostics.conservation_error"],
            torch.zeros(()),
        )
        
    def test_non_negative_constraint(self):
        """Test non-negative concentration enforcement."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestNonNeg',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,),
            non_negative=True
        )
        
        # Create state that could go negative
        concentration = torch.ones(1, 10) * 1.0  # Very small values
        concentration[0, 5] = 0.01  # Even smaller
        
        # Large sink
        source = torch.zeros(1, 10)
        source[0, 5] = -10.0
        
        state = {
            "tracer_concentration": concentration,
            "source_term": source
        }
        
        with pytest.raises(ValueError, match="non-negative"):
            diff.forward(state, time=0.0, dt=0.01)

    def test_source_term_must_be_tensor(self):
        diff = FickianDiffusion(
            spatial_dims=1,
            diffusivity=0.1,
            grid_spacing=(1.0,),
        )
        with pytest.raises(TypeError, match="source_term"):
            diff.forward(
                {"tracer_concentration": torch.ones(1, 4), "source_term": 0.1},
                time=0.0,
                dt=0.1,
            )

    def test_concentration_must_be_tensor(self):
        diff = FickianDiffusion(
            spatial_dims=1,
            diffusivity=0.1,
            grid_spacing=(1.0,),
        )
        with pytest.raises(TypeError, match="tracer_concentration"):
            diff.forward(
                {"tracer_concentration": [[1.0, 2.0, 3.0, 4.0]]},
                time=0.0,
                dt=0.1,
            )

    def test_no_secondary_rate_or_boundary_configuration_path(self):
        diff = FickianDiffusion(
            spatial_dims=1,
            diffusivity=0.1,
            grid_spacing=(1.0,),
        )
        assert not hasattr(diff, "_compute_rate_core")
        assert not hasattr(diff, "set_boundary_conditions")
        assert not hasattr(diff, "auto_adjust_timestep")


class TestUtilityFunctions:
    """Test utility functions for creating diffusion."""
    
    def test_create_isotropic_diffusion(self):
        """Test creation of isotropic diffusion."""
        diff = create_isotropic_diffusion(
            spatial_dims=2,
            diffusivity=1e-5,
            grid_spacing=(0.01, 0.01)
        )
        
        assert isinstance(diff, FickianDiffusion)
        assert diff.spatial_dims == 2
        assert diff.isotropic
        assert diff.diffusivity_scalar == 1e-5
        
    def test_create_anisotropic_diffusion(self):
        """Test creation of anisotropic diffusion."""
        diff = create_anisotropic_diffusion(
            spatial_dims=3,
            diffusivity_components=(1e-5, 2e-5, 3e-5),
            grid_spacing=(0.01, 0.01, 0.01)
        )
        
        assert isinstance(diff, FickianDiffusion)
        assert diff.spatial_dims == 3
        assert not diff.isotropic
        assert diff.diffusivity_tensor is not None


class TestTemperaturePressureCorrections:
    """Test diffusivity corrections for temperature and pressure."""
    
    def test_temperature_correction(self):
        """Test temperature-dependent diffusivity."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestTempCorr',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        # Get diffusivity at different temperatures
        D_ref = diff.get_diffusivity()  # Reference
        D_hot = diff.get_diffusivity(temperature=torch.tensor(350.0))  # Hot
        D_cold = diff.get_diffusivity(temperature=torch.tensor(250.0))  # Cold
        
        # Hot temperature should increase diffusivity
        assert D_hot > D_ref
        # Cold temperature should decrease diffusivity
        assert D_cold < D_ref
        
    def test_pressure_correction(self):
        """Test pressure-dependent diffusivity."""
        config = {'parameters': {'diffusivity': 1e-5}}
        diff = FickianDiffusion(
            name='TestPressCorr',
            config=config,
            spatial_dims=1,
            diffusivity=1e-5,
            grid_spacing=(0.01,)
        )
        
        # Get diffusivity at different pressures
        D_ref = diff.get_diffusivity()  # Reference (1 atm)
        D_high_P = diff.get_diffusivity(pressure=torch.tensor(202650.0))  # 2 atm
        D_low_P = diff.get_diffusivity(pressure=torch.tensor(50662.5))  # 0.5 atm
        
        # High pressure should decrease diffusivity (for gases)
        assert D_high_P < D_ref
        # Low pressure should increase diffusivity
        assert D_low_P > D_ref


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



class TestStrictRuntimePoint:
    """Level 4 dPH runtime point of the closed finite-volume diffusion law."""

    @staticmethod
    def _functor(**overrides):
        kwargs = dict(
            spatial_dims=1,
            diffusivity=torch.tensor([1.0, 4.0, 1.0], dtype=torch.float64),
            diffusivity_layout="scalar_field",
            grid_spacing=1.0,
            boundary_method="zero",
            track_conservation=False,
            strict_runtime_point=True,
        )
        kwargs.update(overrides)
        return FickianDiffusion(**kwargs)

    def test_strict_point_audit_matches_fv_rate_and_quadratic_dissipation(self):
        # References: van der Schaft and Jeltsema (2014); Patankar (1980).
        # Mathematical expression: H = 0.5 c.c, e_H = c, R = d^T diag(K_f) d / dx^2,
        #   x_dot = -R c = FV rate, dH/dt = -c^T R c = -sigma, sigma = sum_f K_f (dc_f/dx)^2.
        # Parameter meanings: K = [1, 4, 1] cell diffusivities, K_f = harmonic mean = 1.6.
        # Units: c kg m-3, K m2 s-1, dx m, sigma kg2 m-6 s-1.
        # Domain/constraints: zero-flux closed boundary, one dimension, no batch.
        # Sign convention: dissipation positive, closed-port energy rate negative.
        functor = self._functor()
        assert functor.last_strict_iphs_audit() is None
        state = {"tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)}
        increments = functor.forward(dict(state), 0.0, 0.125)
        reference = self._functor(strict_runtime_point=False).forward(dict(state), 0.0, 0.125)
        torch.testing.assert_close(
            increments["tracer_concentration"].value, reference["tracer_concentration"].value, rtol=0.0, atol=0.0
        )
        audit = functor.last_strict_iphs_audit()
        assert audit is not None
        assert audit["required_level"] == 4
        assert audit["runtime_verified_level"] == 4
        assert audit["process_level_status"] == "runtime_point_verified"
        assert audit["formulation"] == "dissipative_ph"
        assert audit["closed_port"] is True
        assert audit["dissipation_power"] == pytest.approx(3.2)
        assert audit["closed_port_energy_rate"] == pytest.approx(-3.2)
        assert audit["power_balance_residual"] == pytest.approx(0.0, abs=1e-12)
        assert audit["fv_rate_residual"] == pytest.approx(0.0, abs=1e-12)
        assert audit["state_size"] == 3

    def test_strict_point_periodic_seam_is_part_of_the_resistance(self):
        functor = self._functor(diffusivity=torch.tensor(1.0, dtype=torch.float64), boundary_method="periodic")
        functor.forward({"tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)}, 0.0, 0.125)
        audit = functor.last_strict_iphs_audit()
        assert audit["dissipation_power"] == pytest.approx(2.0)
        assert audit["closed_port_energy_rate"] == pytest.approx(-2.0)
        assert audit["fv_rate_residual"] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("overrides", [
        {"boundary_method": "dirichlet"},
        {"boundary_method": "extrapolate"},
        {"spatial_dims": 2, "diffusivity": torch.tensor(1.0, dtype=torch.float64), "grid_spacing": [1.0, 1.0]},
    ])
    def test_strict_point_selection_rejects_open_or_multidimensional_configurations(self, overrides):
        with pytest.raises(ValueError, match="strict_runtime_point"):
            self._functor(**overrides)

    def test_strict_point_rejects_a_nonzero_source_term_before_integrating_it(self):
        # The audited closed dPH drift is source-free (x_dot = -R e_H); a nonzero
        # source would be integrated by the production step outside that audit.
        functor = self._functor()
        functor.forward({"tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)}, 0.0, 0.125)
        assert functor.last_strict_iphs_audit() is not None
        state = {
            "tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            "source_term": torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64),
        }
        with pytest.raises(ValueError, match="strict_runtime_point.*source_term"):
            functor.forward(state, 0.0, 0.125)
        assert functor.last_strict_iphs_audit() is None
        from relic.core.functor.functor_exceptions import ExecutionError

        # The governed FunctorBase call reports the same refusal as an ExecutionError.
        with pytest.raises(ExecutionError, match="strict_runtime_point.*source_term"):
            functor(dict(state), 0.0, 0.125)
        assert functor.last_strict_iphs_audit() is None

    def test_strict_point_accepts_an_all_zero_source_term(self):
        state = {"tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)}
        functor = self._functor()
        with_zero = functor.forward(
            dict(state, source_term=torch.zeros(3, dtype=torch.float64)), 0.0, 0.125
        )
        assert functor.last_strict_iphs_audit()["runtime_verified_level"] == 4
        without = self._functor().forward(dict(state), 0.0, 0.125)
        torch.testing.assert_close(
            with_zero["tracer_concentration"].value, without["tracer_concentration"].value, rtol=0.0, atol=0.0
        )

    def test_non_strict_selection_still_integrates_a_source_term(self):
        state = {
            "tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            "source_term": torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64),
        }
        increments = self._functor(strict_runtime_point=False).forward(state, 0.0, 0.125)
        # Closed boundary: mass change equals dt * sum(S) = 0.125 * 0.5.
        assert float(increments["tracer_concentration"].value.sum()) == pytest.approx(0.0625, abs=1e-12)

    def test_strict_point_rejects_batched_states_and_clears_the_audit(self):
        functor = self._functor()
        functor.forward({"tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)}, 0.0, 0.125)
        assert functor.last_strict_iphs_audit() is not None
        with pytest.raises(ValueError, match="single unbatched"):
            functor.forward({"tracer_concentration": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)}, 0.0, 0.125)
        assert functor.last_strict_iphs_audit() is None

    def test_strict_point_runs_inside_the_governed_simulator(self):
        from relic.core.simulator import RELICSimulator

        functor = self._functor(diffusivity=torch.tensor(0.25, dtype=torch.float64))
        simulator = RELICSimulator({"device": "cpu", "conservation_checks": False, "enable_auto_scale": False})
        simulator.initialize_state({"tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)})
        simulator.register_functor(functor, execution_group="fast")
        simulator.step(dt=0.125)
        first = functor.last_strict_iphs_audit()
        simulator.step(dt=0.125)
        second = functor.last_strict_iphs_audit()
        assert first["runtime_verified_level"] == second["runtime_verified_level"] == 4
        assert first["dissipation_power"] > second["dissipation_power"] > 0.0
        assert float(simulator.state_manager.get_state("tracer_concentration").sum()) == pytest.approx(1.0)
        assert simulator.step_count == 2

    def test_strict_point_level4_certificate_validates_with_governed_witness(self):
        from relic.computation.operators.iphs_certificate import load_iphs_equivalence_certificate
        from relic.core.simulator import RELICSimulator

        certificate = load_iphs_equivalence_certificate(
            Path(__file__).resolve().parents[2]
            / "docs" / "source" / "theory" / "iphs_certificates"
            / "transport_diffusion_fickian_diffusion_strict_runtime_point.yaml"
        )
        functor = self._functor(diffusivity=torch.tensor(0.25, dtype=torch.float64))
        assert certificate.iphs_level == 4
        assert certificate.formulation == "dissipative_ph"
        assert certificate.process_id == "relic.scientific.transport.diffusion.FickianDiffusion"
        assert certificate.parameter_keys == functor.required_parameters()
        assert certificate.strict_tolerance_parameter_keys() == ("validation.iphs.runtime_point_tolerance",)

        descriptive = certificate.validate()
        assert descriptive["passed"] is False
        assert descriptive["runtime_binding"] == "descriptive"
        assert any("process_bound_runtime_unavailable" in issue for issue in descriptive["issues"])

        simulator = RELICSimulator({"device": "cpu", "conservation_checks": False, "enable_auto_scale": False})
        simulator.initialize_state({"tracer_concentration": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)})
        simulator.register_functor(functor, execution_group="fast")
        simulator.step(dt=0.125)
        audit = functor.last_strict_iphs_audit()
        assert audit is not None
        provenance = simulator.get_variable_provenance("tracer_concentration")
        assert provenance[-1]["functor"] == functor.name
        audit["governed_execution"] = {
            "entry": "FunctorBase.__call__",
            "functor": functor.name,
            "committed_variables": ["tracer_concentration"],
            "step": simulator.step_count,
        }
        verified = certificate.validate(runtime_point_audit=audit)
        assert verified["issues"] == []
        assert verified["passed"] is True
        assert verified["runtime_verified_level"] == 4
        assert verified["runtime_binding"] == "process_bound"

        # The class-level certificate stays level 3 and untouched by the selection.
        base = load_iphs_equivalence_certificate(
            Path(__file__).resolve().parents[2]
            / "docs" / "source" / "theory" / "iphs_certificates"
            / "transport_diffusion_fickian_diffusion.yaml"
        )
        assert base.iphs_level == 3


class TestDiffusivityLayoutAndBaseScalar:
    """A tensor diffusivity declares its layout; only a scalar D sets the base scalar."""

    @staticmethod
    def _functor(diffusivity, **kwargs):
        options = dict(
            spatial_dims=2, diffusivity=diffusivity, grid_spacing=(0.5, 0.25),
            boundary_method="zero", non_negative=False, track_conservation=False,
        )
        options.update(kwargs)
        return FickianDiffusion(**options)

    def test_tensor_diffusivity_without_a_declared_layout_is_rejected(self):
        from relic.scientific.transport.diffusion import DiffusivityError

        with pytest.raises(DiffusivityError, match="diffusivity_layout"):
            self._functor(torch.tensor([1.0e-5, 1.0e-4], dtype=torch.float64))
        functor = self._functor(1.0e-5)
        assert functor.diffusivity_layout == "scalar_field"
        with pytest.raises(DiffusivityError, match="diffusivity_layout"):
            functor.set_diffusivity(torch.full((1, 3, 2), 1.0e-5, dtype=torch.float64))
        with pytest.raises(ValueError, match="diffusivity_layout"):
            functor.set_diffusivity(torch.tensor([1.0e-5, 1.0e-4]), layout="diagonal")

    @pytest.mark.parametrize("layout", ["directional", "scalar_field"])
    def test_declared_layout_selects_the_reading_of_an_ambiguous_shape(self, layout):
        # References: Patankar (1980) finite-volume diffusion.
        # Mathematical expression: dc/dt = sum_k d/dx_k(K_k dc/dx_k).
        # Parameter meanings: on a (1, 3, 2) grid K = (a, b) is either (Ky, Kx)
        #   ("directional") or an x-profile of one scalar field ("scalar_field").
        # Units: K [m2 s-1], c [kg m-3], dx [m], dt [s].
        # Domain/constraints: both readings fit the shape, so the layout decides.
        # Sign convention: increments are c_next - c.
        from relic.computation.operators.functional import apply_diffusion

        coefficient = torch.tensor([1.0e-2, 4.0e-2], dtype=torch.float64)
        concentration = torch.tensor([[[1.0, 2.0], [4.0, 3.0], [0.5, 5.0]]], dtype=torch.float64)
        functor = self._functor(coefficient, diffusivity_layout=layout)
        assert functor.diffusivity_layout == layout
        delta = _tracer_increment(functor.forward({functor.tracer_key: concentration}, time=0.0, dt=0.1))
        expected = apply_diffusion(
            concentration, coefficient, (0.5, 0.25), 2, "zero", 0.1, True, False,
            run_conservation_check=False, time_integration_method="euler",
            diffusivity_layout=layout,
        )["concentration"] - concentration
        torch.testing.assert_close(delta, expected)
        other = "scalar_field" if layout == "directional" else "directional"
        different = apply_diffusion(
            concentration, coefficient, (0.5, 0.25), 2, "zero", 0.1, True, False,
            run_conservation_check=False, time_integration_method="euler",
            diffusivity_layout=other,
        )["concentration"] - concentration
        assert not torch.allclose(delta, different)

    def test_scalar_float_diffusivity_sets_the_base_scalar(self):
        functor = self._functor(2.0e-5)
        assert functor.diffusivity == pytest.approx(2.0e-5)
        assert isinstance(functor.diffusivity, float)
        field = torch.full((1, 3, 2), 1.0e-5, dtype=torch.float64)
        resolved = functor._resolve_diffusivity_tensor(field)
        assert resolved.dtype == field.dtype and float(resolved) == pytest.approx(2.0e-5)

    def test_learnable_scalar_diffusivity_is_stored_as_the_same_object(self):
        coefficient = torch.nn.Parameter(torch.tensor(1.0e-2, dtype=torch.float64))
        functor = self._functor(coefficient)
        assert functor.diffusivity is coefficient
        concentration = torch.tensor([[[1.0, 2.0], [4.0, 3.0], [0.5, 5.0]]], dtype=torch.float64)
        delta = _tracer_increment(functor.forward({functor.tracer_key: concentration}, time=0.0, dt=0.1))
        delta.pow(2).sum().backward()
        assert coefficient.grad is not None and bool(torch.isfinite(coefficient.grad))
        with torch.no_grad():
            coefficient.mul_(2.0)
        assert float(functor.diffusivity) == pytest.approx(2.0e-2)

    def test_parameter_bound_scalar_sets_the_base_scalar(self, monkeypatch):
        manager = UnifiedParameterManager(device="cpu")
        key = "transport.diffusion.layout_bound_test"
        manager.register_parameter(ParameterSpec(
            key=key, value=0.25, units="m2 s-1", dtype="float64", bounds=(0.0, 1.0),
            requires_grad=True, references=["Crank (1975), The Mathematics of Diffusion"], category="model_parameter",
        ))
        monkeypatch.setattr(functor_base_module, "_get_parameter_manager", lambda: manager)
        functor = FickianDiffusion(
            spatial_dims=1, diffusivity_parameter_key=key, grid_spacing=[1.0],
            boundary_method="zero", non_negative=False, track_conservation=False,
        )
        assert isinstance(functor.diffusivity, torch.Tensor)
        assert functor.diffusivity.requires_grad
        assert float(functor.diffusivity) == pytest.approx(0.25)
        assert functor.diffusivity_layout == "scalar_field"

    @pytest.mark.parametrize(
        ("coefficient", "layout"),
        [
            (torch.tensor([1.0e-2, 4.0e-2], dtype=torch.float64), "directional"),
            (torch.full((1, 3, 2), 1.0e-2, dtype=torch.float64), "scalar_field"),
        ],
    )
    def test_field_diffusivity_leaves_the_base_scalar_none(self, coefficient, layout):
        functor = self._functor(coefficient, diffusivity_layout=layout)
        assert functor.diffusivity is None
        with pytest.raises(RuntimeError, match="has no diffusivity"):
            functor._resolve_diffusivity_tensor(torch.ones(1, 3, 2, dtype=torch.float64))
        functor.set_diffusivity(3.0e-5)
        assert functor.diffusivity == pytest.approx(3.0e-5)
        assert functor.diffusivity_layout == "scalar_field"
        functor.set_diffusivity(coefficient, layout=layout)
        assert functor.diffusivity is None
        assert functor.diffusivity_layout == layout

    def test_anisotropic_factory_declares_the_directional_layout(self):
        functor = create_anisotropic_diffusion(
            spatial_dims=2, diffusivity_components=(1.0e-5, 2.0e-5), grid_spacing=(0.01, 0.01)
        )
        assert functor.diffusivity_layout == "directional"
        assert functor.diffusivity is None
