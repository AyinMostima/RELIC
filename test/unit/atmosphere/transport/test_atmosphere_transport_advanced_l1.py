"""
L1 Unit Tests for Advanced Atmosphere Transport Modules

Comprehensive testing of advanced atmospheric transport mechanisms including
dry deposition aggregation and eddy diffusion boundary layer models.

Test Coverage:
- Basic instantiation and parameter initialization
- Dry deposition aggregation mechanisms
- Size-dependent deposition velocities
- Surface resistance calculations
- Eddy diffusion boundary layer dynamics
- Turbulent transport calculations
- Stability corrections and surface flux parametrizations
- Environmental response functions
- Parameter sensitivity analysis
- Boundary conditions and robustness
- Physical accuracy validation against observations

Scientific Validation:
- Deposition velocity: v_d = 1/(r_a + r_b + r_c) where r = resistances
- Eddy diffusivity: K_z = kappa*u_star*z/phi_h(z/L)
- Reduced box exchange: DeltaC = dt*K_z/h^2*(C_background-C)
- Stability parameter: zeta = z/L where L is Monin-Obukhov length
- Resistance components: r_a (aerodynamic), r_b (boundary layer), r_c (surface)

References:
- Wesely (1989). Parameterization of surface resistances to gaseous dry deposition
- Zhang et al. (2001). A size-segregated particle dry deposition scheme
- Businger et al. (1971). Flux-profile relationships in atmospheric surface layer
- Dyer & Hicks (1970). Flux-gradient relationships in constant flux layer
- Baldocchi et al. (1987). Measuring biosphere-atmosphere exchanges of biologically related gases
"""

import torch
import pytest

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from relic.scientific.atmosphere.transport.dry_deposition_aggregator import (
    DryDepositionAggregator,
)
from relic.scientific.atmosphere.transport.eddy_diffusion_blm import (
    EddyDiffusionBLMFunctor,
    EddyDiffusionConfig,
)
from relic.core.state import UpdateItem, UpdateMode


class TestDryDepositionAggregatorL1:
    """Advanced L1 tests for the canonical normalized boundary sink."""

    @pytest.fixture
    def deposition_model(self):
        """Create the production Functor without numerical config overrides."""
        return DryDepositionAggregator(name="test_deposition")

    @staticmethod
    def canonical_state(**overrides):
        """Return a complete canonical dry-deposition state."""
        state = {
            "no_mixing_ratio": torch.tensor(10.0e-9),
            "no2_mixing_ratio": torch.tensor(20.0e-9),
            "o3_mixing_ratio": torch.tensor(50.0e-9),
            "so2_mixing_ratio": torch.tensor(2.0e-9),
            "pbl_height": torch.tensor(1000.0),
            "aerodynamic_resistance": torch.tensor(50.0),
        }
        state.update(overrides)
        return state

    def test_basic_instantiation_and_honest_metadata(self, deposition_model):
        """Metadata identifies a normalized non-strict state increment."""
        assert isinstance(deposition_model, DryDepositionAggregator)
        assert deposition_model.get_dependencies() == ["atmospheric_turbulence"]
        assert deposition_model.metadata.variable_kind.value == "state"
        assert set(deposition_model.metadata.conservation_laws) == {"mass"}

        metadata = deposition_model.get_transport_metadata()
        assert metadata["closure"] == "resistance_law"
        assert metadata["strict_surface_flux_proof"] is False
        assert metadata["normalization_assumption"] == (
            "well_mixed_pbl_with_constant_air_molar_density"
        )
        assert metadata["open_system_route"] == "boundary_export"

    def test_arbitrary_spatial_dimensions_broadcast(self, deposition_model):
        """Spatial fields broadcast without assuming a fixed rank."""
        concentration = torch.tensor(
            [[10.0e-9, 20.0e-9, 30.0e-9], [40.0e-9, 50.0e-9, 60.0e-9]]
        )
        state = {
            "no_mixing_ratio": concentration,
            "no2_mixing_ratio": concentration * 2.0,
            "o3_mixing_ratio": concentration * 5.0,
            "so2_mixing_ratio": concentration * 0.2,
            "pbl_height": torch.tensor([[500.0, 1000.0, 1500.0]]),
            "aerodynamic_resistance": torch.tensor([[20.0], [80.0]]),
        }

        output = deposition_model.forward(state, time=0.0, dt=300.0)

        for species in deposition_model.target_species:
            key = deposition_model.species_map[species]
            increment = output[key].value
            assert increment.shape == (2, 3)
            assert torch.isfinite(increment).all()
            assert torch.all(increment < 0.0)

    def test_differentiability_is_preserved(self, deposition_model):
        """The canonical resistance calculation preserves autograd."""
        state = self.canonical_state()
        for species in deposition_model.target_species:
            key = deposition_model.species_map[species]
            state[key] = state[key].clone().requires_grad_(True)

        output = deposition_model.forward(state, time=0.0, dt=60.0)
        loss = sum(
            output[deposition_model.species_map[species]].value.sum()
            for species in deposition_model.target_species
        )
        loss.backward()

        for species in deposition_model.target_species:
            key = deposition_model.species_map[species]
            assert state[key].grad is not None
            assert torch.isfinite(state[key].grad).all()
            assert torch.all(state[key].grad < 0.0)

    def test_dt_scaling_before_availability_limit(self, deposition_model):
        """Unclipped state increments scale linearly with timestep."""
        state = self.canonical_state()

        one_minute = deposition_model.forward(state, time=0.0, dt=60.0)
        two_minutes = deposition_model.forward(state, time=0.0, dt=120.0)

        for species in deposition_model.target_species:
            key = deposition_model.species_map[species]
            assert torch.allclose(
                two_minutes[key].value,
                2.0 * one_minute[key].value,
                rtol=1.0e-6,
            )

    def test_upstream_resistance_must_be_on_canonical_state(self, deposition_model):
        """The declared turbulence output is read from the canonical state only: it is
        required there, and a larger state resistance lowers the deposition."""
        state = self.canonical_state()
        del state["aerodynamic_resistance"]

        with pytest.raises(KeyError, match="aerodynamic_resistance"):
            deposition_model.forward(state, time=0.0, dt=60.0)

        low = deposition_model.forward(
            self.canonical_state(aerodynamic_resistance=torch.tensor(50.0)), time=0.0, dt=60.0
        )
        high = deposition_model.forward(
            self.canonical_state(aerodynamic_resistance=torch.tensor(500.0)), time=0.0, dt=60.0
        )
        for species in deposition_model.target_species:
            key = deposition_model.species_map[species]
            assert torch.all(torch.abs(high[key].value) < torch.abs(low[key].value))

    def test_invalid_activation_type_fails_fast(self):
        """String activation values cannot be interpreted by Python truthiness."""
        from relic.scientific.atmosphere.transport.dry_deposition_aggregator import (
            DryDepositionConfig,
        )

        with pytest.raises(TypeError, match="enabled must be bool"):
            DryDepositionAggregator(
                deposition_config=DryDepositionConfig(
                    enabled="false",  # type: ignore[arg-type]
                )
            )

    def test_phase8_hodge_is_declared_reference_not_runtime_flux(
        self,
        deposition_model,
    ):
        """The level-3 Hodge remains positive and explicitly structural."""
        hodge = deposition_model.build_hodge_contract()
        metadata = deposition_model.get_transport_metadata()

        assert torch.all(torch.diag(hodge.matrix) > 0.0)
        assert (
            "atmosphere.dry_deposition."
            "structural_reference_deposition_velocity"
            in hodge.parameter_keys
        )
        assert metadata["strict_surface_flux_proof"] is False


class TestEddyDiffusionBLML1:
    """Advanced L1 tests for the canonical EddyDiffusionBLMFunctor."""
    
    @pytest.fixture
    def eddy_diffusion_model(self):
        """Create the real production functor with the canonical species schema."""
        return EddyDiffusionBLMFunctor(
            name="advanced_eddy_diffusion",
            transport_config=EddyDiffusionConfig(
                enabled=True,
            ),
        )

    @staticmethod
    def canonical_state(**overrides):
        """Return a complete canonical Eddy input state."""
        state = {
            "no_mixing_ratio": torch.tensor(10.0e-9),
            "no2_mixing_ratio": torch.tensor(20.0e-9),
            "o3_mixing_ratio": torch.tensor(50.0e-9),
            "isoprene_mixing_ratio": torch.tensor(2.0e-9),
            "pbl_height": torch.tensor(800.0),
            "friction_velocity": torch.tensor(0.3),
            "monin_obukhov_length": torch.tensor(-150.0),
        }
        state.update(overrides)
        return state
    
    def test_basic_instantiation(self, eddy_diffusion_model):
        """Test L1.1: Basic instantiation and parameter initialization."""
        assert isinstance(eddy_diffusion_model, EddyDiffusionBLMFunctor)
        assert eddy_diffusion_model.get_dependencies() == ["atmospheric_turbulence"]
        assert eddy_diffusion_model.metadata.input_variables == [
            "no_mixing_ratio",
            "no2_mixing_ratio",
            "o3_mixing_ratio",
            "isoprene_mixing_ratio",
            "pbl_height",
            "friction_velocity",
            "monin_obukhov_length",
        ]
        assert not hasattr(eddy_diffusion_model, "set_transport_mode")
    
    def test_eddy_diffusivity_calculations(self, eddy_diffusion_model):
        """Test vectorized K-theory increments and differentiability."""
        no = torch.tensor([10.0e-9, 12.0e-9], requires_grad=True)
        state = self.canonical_state(
            no_mixing_ratio=no,
            no2_mixing_ratio=torch.tensor([20.0e-9, 25.0e-9]),
            o3_mixing_ratio=torch.tensor([50.0e-9, 55.0e-9]),
            isoprene_mixing_ratio=torch.tensor([2.0e-9, 3.0e-9]),
            pbl_height=torch.tensor([800.0, 1000.0]),
            friction_velocity=torch.tensor([0.3, 0.4]),
            monin_obukhov_length=torch.tensor([-150.0, 200.0]),
        )
        results = eddy_diffusion_model.forward(state, time=0.0, dt=60.0)
        k_value = results["diagnostics.eddy_diffusion.eddy_diffusivity"]
        k_min, k_max = eddy_diffusion_model.get_transport_metadata()[
            'parameter_snapshot'
        ]['k_bounds']
        assert k_value.shape == no.shape
        assert torch.all(k_value >= k_min)
        assert torch.all(k_value <= k_max)
        no_update = results["no_mixing_ratio"]
        assert isinstance(no_update, UpdateItem)
        no_update.value.sum().backward()
        assert no.grad is not None
        assert torch.isfinite(no.grad).all()
    
    def test_height_dependence(self, eddy_diffusion_model):
        """Test one-box midpoint geometry across PBL heights."""
        heights = [200.0, 500.0, 1000.0, 2000.0]
        diffusivities = []
        for height in heights:
            state = self.canonical_state(
                pbl_height=torch.tensor(height),
                monin_obukhov_length=torch.tensor(1.0e6),
            )
            result = eddy_diffusion_model.forward(state, time=0.0, dt=60.0)
            diffusivities.append(
                result["diagnostics.eddy_diffusion.eddy_diffusivity"].item()
            )
        assert diffusivities == sorted(diffusivities)
        k_min, k_max = eddy_diffusion_model.get_transport_metadata()[
            'parameter_snapshot'
        ]['k_bounds']
        assert all(
            k_min <= value <= k_max
            for value in diffusivities
        )
    
    def test_stability_corrections(self, eddy_diffusion_model):
        """Test L1.4: Atmospheric stability corrections."""
        values = {}
        for condition, length in (
            ("unstable", -200.0),
            ("neutral", 1.0e6),
            ("stable", 200.0),
        ):
            state = self.canonical_state(
                pbl_height=torch.tensor(200.0),
                friction_velocity=torch.tensor(0.1),
                monin_obukhov_length=torch.tensor(length),
            )
            result = eddy_diffusion_model.forward(state, time=0.0, dt=10.0)
            values[condition] = result[
                "diagnostics.eddy_diffusion.eddy_diffusivity"
            ].item()
        assert values["unstable"] > values["neutral"] > values["stable"]
    
    def test_friction_velocity_scaling(self, eddy_diffusion_model):
        """Test L1.5: Friction velocity scaling relationships."""
        friction_velocities = [0.05, 0.1, 0.2, 0.4]
        diffusivities = []
        for u_star in friction_velocities:
            state = self.canonical_state(
                pbl_height=torch.tensor(200.0),
                friction_velocity=torch.tensor(u_star),
                monin_obukhov_length=torch.tensor(1.0e6),
            )
            result = eddy_diffusion_model.forward(state, time=0.0, dt=10.0)
            diffusivities.append(
                result["diagnostics.eddy_diffusion.eddy_diffusivity"].item()
            )
        reference = diffusivities[0]
        for u_star, diffusivity in zip(friction_velocities, diffusivities):
            expected_ratio = u_star / friction_velocities[0]
            assert diffusivity / reference == pytest.approx(expected_ratio, rel=2e-3)
    
    def test_boundary_layer_height_effects(self, eddy_diffusion_model):
        """Test L1.6: Boundary layer height effects on mixing."""
        bl_heights = [200.0, 500.0, 1000.0]
        relaxation_rates = []
        for bl_height in bl_heights:
            state = self.canonical_state(
                pbl_height=torch.tensor(bl_height),
                friction_velocity=torch.tensor(0.1),
                monin_obukhov_length=torch.tensor(1.0e6),
            )
            result = eddy_diffusion_model.forward(state, time=0.0, dt=10.0)
            diffusivity = result[
                "diagnostics.eddy_diffusion.eddy_diffusivity"
            ].item()
            relaxation_rates.append(diffusivity / bl_height**2)
        assert relaxation_rates == sorted(relaxation_rates, reverse=True)
    
    def test_turbulent_flux_calculations(self, eddy_diffusion_model):
        """Test the signed top-boundary exchange ledger by species."""
        defaults = eddy_diffusion_model._transport_parameters
        state = self.canonical_state(
            no_mixing_ratio=torch.tensor(
                2.0 * defaults["background_no_mixing_ratio"]
            ),
            no2_mixing_ratio=torch.tensor(
                0.5 * defaults["background_no2_mixing_ratio"]
            ),
            o3_mixing_ratio=torch.tensor(
                defaults["background_o3_mixing_ratio"]
            ),
            isoprene_mixing_ratio=torch.tensor(0.0),
        )
        result = eddy_diffusion_model.forward(state, time=0.0, dt=60.0)
        assert result["no_mixing_ratio"].value.item() < 0.0
        assert result["no2_mixing_ratio"].value.item() > 0.0
        assert result["o3_mixing_ratio"].value.item() == pytest.approx(0.0)
        assert result["isoprene_mixing_ratio"].value.item() > 0.0
    
    def test_numerical_stability(self, eddy_diffusion_model):
        """Test L1.8: Numerical stability under extreme conditions."""
        extreme_states = [
            self.canonical_state(
                pbl_height=torch.tensor(50.0),
                friction_velocity=torch.tensor(0.01),
                monin_obukhov_length=torch.tensor(-5.0),
            ),
            self.canonical_state(
                pbl_height=torch.tensor(3000.0),
                friction_velocity=torch.tensor(1.5),
                monin_obukhov_length=torch.tensor(1000.0),
            ),
            self.canonical_state(
                pbl_height=torch.tensor(1000.0),
                friction_velocity=torch.tensor(0.25),
                monin_obukhov_length=torch.tensor(1.0e8),
            ),
        ]
        for state in extreme_states:
            result = eddy_diffusion_model.forward(state, time=0.0, dt=1.0)
            assert torch.isfinite(
                result["diagnostics.eddy_diffusion.eddy_diffusivity"]
            ).all()
            for key in (
                "no_mixing_ratio",
                "no2_mixing_ratio",
                "o3_mixing_ratio",
                "isoprene_mixing_ratio",
            ):
                assert torch.all(state[key] + result[key].value >= 0.0)

        unstable_step = self.canonical_state(
            pbl_height=torch.tensor(50.0),
            friction_velocity=torch.tensor(2.0),
            monin_obukhov_length=torch.tensor(1.0e8),
        )
        with pytest.raises(RuntimeError, match="stability condition violated"):
            eddy_diffusion_model.forward(unstable_step, time=0.0, dt=60.0)
    
    def test_forward_interface_consistency(self, eddy_diffusion_model):
        """Test exact metadata, runtime keys, and update semantics."""
        result = eddy_diffusion_model.forward(
            self.canonical_state(),
            time=0.0,
            dt=60.0,
        )
        assert set(result) == set(eddy_diffusion_model.metadata.output_variables)
        for key in (
            "no_mixing_ratio",
            "no2_mixing_ratio",
            "o3_mixing_ratio",
            "isoprene_mixing_ratio",
        ):
            item = result[key]
            assert isinstance(item, UpdateItem)
            assert item.mode is UpdateMode.ACCUMULATE
            assert item.per_timestep is True
            assert item.unit == "mol mol-1"
        for key, value in result.items():
            if key.startswith("diagnostics."):
                assert isinstance(value, torch.Tensor)
                assert torch.isfinite(value).all()


if __name__ == "__main__":
    pytest.main([__file__])



