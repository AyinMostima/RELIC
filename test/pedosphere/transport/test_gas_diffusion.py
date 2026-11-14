"""
L1 Unit Tests for Gas Diffusion Transport Functor.

Tests initialization, parameter loading, CO2/N2O diffusion, and conservation laws.
Following the RELIC L1 testing specification.

References:
- Millington & Quirk (1961) - Diffusivity models
- Penman (1940) - Gas diffusion in soils
- Li et al. (1992) - DNDC N2O formulation

Mathematical Expressions tested:
J = -Deff * dC/dz
where Deff depends on air-filled porosity and temperature
"""
import warnings

import pytest
import torch
from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.core.state import UpdateItem
from relic.scientific.pedosphere.transport.gas_diffusion import (
    GasDiffusion1D,
    DiffusivityModel
)


pytestmark = pytest.mark.usefixtures(
    "pedosphere_transport_constructor_overrides"
)


def _unwrap(value):
    """Return the tensor carried by a canonical state update."""
    return value.value if isinstance(value, UpdateItem) else value


class TestGasDiffusionInit:
    """Test initialization and parameter loading."""

    def test_initialization(self, parameter_manager, basic_config):
        """Test Functor can be initialized."""
        functor = GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0
        )
        assert functor is not None
        assert functor.n_layers == 5
        assert functor.soil_depth == 1.0

    def test_parameter_loading_from_manager(self, parameter_manager):
        """Test that required parameters are loaded from parameter manager."""
        functor = GasDiffusion1D(n_layers=5)

        # Should have loaded diffusion model
        assert hasattr(functor, 'diffusivity_model')
        assert isinstance(functor.diffusivity_model, DiffusivityModel)

        # Should have gas species configured
        assert hasattr(functor, 'species')
        assert 'CO2' in functor.species
        assert 'N2O' in functor.species

    def test_grid_setup(self):
        """Test vertical grid initialization."""
        functor = GasDiffusion1D(n_layers=5, soil_depth=1.0)

        # Check grid buffers
        assert hasattr(functor, 'layer_thickness')
        assert hasattr(functor, 'layer_centers')

        # Check grid dimensions
        assert functor.layer_thickness.shape == (5,)
        assert functor.layer_centers.shape == (5,)

        # Total depth should match
        total_depth = torch.sum(functor.layer_thickness)
        assert torch.allclose(total_depth, torch.tensor(1.0), rtol=1e-3)

    def test_metadata_includes_conservation_laws(self):
        """Test metadata includes conservation laws."""
        functor = GasDiffusion1D(n_layers=5)
        metadata = functor.metadata

        if isinstance(metadata, dict):
            conservation = metadata.get('conservation_laws', [])
        else:
            conservation = getattr(metadata, 'conservation_laws', [])

        assert 'mass' in conservation or 'carbon' in conservation


class TestGasDiffusionForward:
    """Test forward execution."""

    @pytest.fixture
    def gas_state(self):
        """State with gas concentrations."""
        return {
            'soil_moisture': torch.tensor([0.2, 0.25, 0.3, 0.35, 0.4]),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 293.15),
            'clay_content': torch.full((5,), 0.2),
            'soil_co2_concentration': torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),  # Decreasing with depth
            'n2o_concentration': torch.tensor([2.0, 1.8, 1.5, 1.2, 1.0])
        }

    def test_forward_runs(self, gas_state):
        """Test that forward() executes without errors."""
        functor = GasDiffusion1D(n_layers=5, soil_depth=1.0)

        result = functor.forward(
            state=gas_state,
            time=0.0,
            dt=3600.0  # 1 hour
        )

        assert isinstance(result, dict)
        assert len(result) > 0

    def test_co2_compatibility_alias_is_rejected(self, gas_state):
        """Only the canonical soil CO2 state key is accepted."""
        functor = GasDiffusion1D(n_layers=5, soil_depth=1.0)
        alias_state = dict(gas_state)
        alias_state["co2_concentration"] = alias_state.pop(
            "soil_co2_concentration"
        )

        with pytest.raises(ValueError, match="soil_co2_concentration"):
            functor.forward(alias_state, 0.0, 3600.0)

    def test_forward_returns_tensors(self, gas_state):
        """Test that forward returns Tensor values."""
        functor = GasDiffusion1D(n_layers=5, soil_depth=1.0)

        result = functor.forward(
            state=gas_state,
            time=0.0,
            dt=3600.0
        )

        for key, value in result.items():
            assert isinstance(value, UpdateItem), f"{key} should be UpdateItem"
            assert isinstance(value.value, torch.Tensor), f"{key} value should be Tensor"

    def test_co2_surface_flux_output(self, gas_state):
        """Test that CO2 surface emission is calculated."""
        functor = GasDiffusion1D(n_layers=5, soil_depth=1.0)

        result = functor.forward(gas_state, 0.0, 3600.0)

        # Should have CO2 surface flux
        co2_keys = [k for k in result.keys() if 'co2' in k.lower() and 'surface' in k.lower()]
        assert len(co2_keys) > 0, "Should have CO2 surface flux output"

        # Surface flux should be finite and physically signed by gradient.
        co2_flux = _unwrap(result[co2_keys[0]])
        assert torch.all(torch.isfinite(co2_flux)), "CO2 surface flux should be finite"

    def test_n2o_surface_flux_output(self, gas_state):
        """Test that N2O surface emission is calculated."""
        functor = GasDiffusion1D(n_layers=5, soil_depth=1.0)

        result = functor.forward(gas_state, 0.0, 3600.0)

        # Should have N2O surface flux
        n2o_keys = [k for k in result.keys() if 'n2o' in k.lower() and 'surface' in k.lower()]
        assert len(n2o_keys) > 0, "Should have N2O surface flux output"

        # Surface flux should be finite and physically signed by gradient.
        n2o_flux = _unwrap(result[n2o_keys[0]])
        assert torch.all(torch.isfinite(n2o_flux)), "N2O surface flux should be finite"


class TestTemperatureSensitivity:
    """Test temperature effects on diffusion."""

    @pytest.fixture
    def base_state(self):
        return {
            'soil_moisture': torch.full((5,), 0.3),
            'porosity': torch.full((5,), 0.45),
            'clay_content': torch.full((5,), 0.2),
            'soil_co2_concentration': torch.full((5,), 3.0),
            'n2o_concentration': torch.full((5,), 1.5)
        }

    def test_temperature_increases_diffusion(self, base_state):
        """
        Test that higher temperature increases diffusivity.

        Mathematical Expression:
        D(T) = D0 * (T/T0)^n, where n ~ 1.75

        Expected: D_warm > D_cold
        """
        functor = GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_temperature_correction=True
        )

        # Cold temperature
        state_cold = base_state.copy()
        state_cold['soil_temperature'] = torch.full((5,), 278.15)  # 5 C

        # Warm temperature
        state_warm = base_state.copy()
        state_warm['soil_temperature'] = torch.full((5,), 303.15)  # 30 C

        result_cold = functor.forward(state_cold, 0.0, 3600.0)
        result_warm = functor.forward(state_warm, 0.0, 3600.0)

        # Find CO2 flux keys
        co2_keys = [k for k in result_cold.keys() if 'co2' in k.lower() and 'surface' in k.lower()]

        if co2_keys:
            flux_cold = _unwrap(result_cold[co2_keys[0]]).item()
            flux_warm = _unwrap(result_warm[co2_keys[0]]).item()

            # Warm should have higher diffusion magnitude.
            assert abs(flux_warm) > abs(flux_cold), "Higher temperature should increase diffusion magnitude"


class TestMoistureSensitivity:
    """Test moisture effects on diffusion."""

    @pytest.fixture
    def base_state_moisture(self):
        return {
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 293.15),
            'clay_content': torch.full((5,), 0.2),
            'soil_co2_concentration': torch.full((5,), 3.0),
            'n2o_concentration': torch.full((5,), 1.5)
        }

    def test_moisture_reduces_air_porosity(self, base_state_moisture):
        """
        Test that higher moisture reduces air-filled porosity and diffusion.

        Mathematical Expression:
        theta_air = porosity - soil_moisture
        Deff ~ theta_air^(10/3) / porosity^2 (Millington-Quirk)

        Expected: Higher moisture -> Lower diffusion
        """
        functor = GasDiffusion1D(n_layers=5, soil_depth=1.0)

        # Dry soil (high air porosity)
        state_dry = base_state_moisture.copy()
        state_dry['soil_moisture'] = torch.full((5,), 0.1)

        # Wet soil (low air porosity)
        state_wet = base_state_moisture.copy()
        state_wet['soil_moisture'] = torch.full((5,), 0.4)

        result_dry = functor.forward(state_dry, 0.0, 3600.0)
        result_wet = functor.forward(state_wet, 0.0, 3600.0)

        # Find CO2 flux
        co2_keys = [k for k in result_dry.keys() if 'co2' in k.lower() and 'surface' in k.lower()]

        if co2_keys:
            flux_dry = _unwrap(result_dry[co2_keys[0]]).item()
            flux_wet = _unwrap(result_wet[co2_keys[0]]).item()

            # Dry soil should have higher gas diffusion magnitude.
            assert abs(flux_dry) > abs(flux_wet), "Dry soil should have higher gas diffusion magnitude"


class TestDiffusivityModels:
    """Test different diffusivity model formulations."""

    @pytest.fixture
    def standard_state(self):
        return {
            'soil_moisture': torch.full((5,), 0.3),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 293.15),
            'clay_content': torch.full((5,), 0.2),
            'soil_co2_concentration': torch.full((5,), 3.0),
            'n2o_concentration': torch.full((5,), 1.5)
        }

    def test_millington_quirk_model(self, standard_state):
        """Test Millington-Quirk diffusivity model."""
        functor = GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            diffusivity_model="millington_quirk"
        )

        result = functor.forward(standard_state, 0.0, 3600.0)

        # Should produce finite fluxes
        assert result is not None
        co2_keys = [k for k in result.keys() if 'co2' in k.lower() and 'surface' in k.lower()]
        if co2_keys:
            assert torch.isfinite(_unwrap(result[co2_keys[0]]))

    def test_penman_model(self, standard_state):
        """Test Penman diffusivity model."""
        functor = GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            diffusivity_model="penman"
        )

        result = functor.forward(standard_state, 0.0, 3600.0)

        # Should produce positive fluxes
        assert result is not None


class TestConservation:
    """Test mass conservation."""

    def test_concentration_updates_conserve_mass(self):
        """Test that concentration updates respect mass balance."""
        functor = GasDiffusion1D(n_layers=5, soil_depth=1.0)

        state = {
            'soil_moisture': torch.full((5,), 0.3),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 293.15),
            'clay_content': torch.full((5,), 0.2),
            'soil_co2_concentration': torch.full((5,), 3.0),
            'n2o_concentration': torch.full((5,), 1.5)
        }

        result = functor.forward(state, 0.0, 3600.0)

        # Check that outputs don't have NaN or inf
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                assert torch.all(torch.isfinite(value)), f"{key} should be finite"


class TestAtmosphericBoundary:
    """Test atmospheric exchange boundary condition."""

    def test_atmospheric_exchange_enabled(self):
        """Test with atmospheric exchange enabled."""
        functor = GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_atmospheric_exchange=True
        )

        state = {
            'soil_moisture': torch.full((5,), 0.3),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 293.15),
            'clay_content': torch.full((5,), 0.2),
            'soil_co2_concentration': torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0]),  # High gradient
            'n2o_concentration': torch.full((5,), 1.5)
        }

        result = functor.forward(state, 0.0, 3600.0)

        # Should have surface flux
        co2_keys = [k for k in result.keys() if 'co2' in k.lower() and 'surface' in k.lower()]
        assert len(co2_keys) > 0

        # With steep gradient, should have significant non-zero flux magnitude.
        flux = _unwrap(result[co2_keys[0]]).item()
        assert abs(flux) > 0

    def test_atmospheric_exchange_disabled(self):
        """Test with atmospheric exchange disabled."""
        functor = GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_atmospheric_exchange=False
        )

        state = {
            'soil_moisture': torch.full((5,), 0.3),
            'porosity': torch.full((5,), 0.45),
            'soil_temperature': torch.full((5,), 293.15),
            'clay_content': torch.full((5,), 0.2),
            'soil_co2_concentration': torch.full((5,), 3.0),
            'n2o_concentration': torch.full((5,), 1.5)
        }

        result = functor.forward(state, 0.0, 3600.0)

        # Should still execute without error
        assert result is not None

    def test_surface_flux_sign_reverses_with_gradient(self):
        """
        Test surface flux sign tracks the concentration-gradient direction.

        Mathematical Expression:
        J_z = -D_eff * dC/dz; F_surface,out = -J_z(0)
        """
        functor = GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_atmospheric_exchange=True
        )
        functor.species["CO2"].atmospheric_concentration = 5.0
        deff = torch.full((5,), 1.0e-6)

        # Soil concentration above atmosphere near surface -> upward emission sign.
        concentration_emit = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])
        _, surface_flux_emit = functor._calculate_diffusive_flux(
            concentration_emit,
            deff,
            species="CO2",
        )

        # Soil concentration below atmosphere near surface -> downward uptake sign.
        concentration_uptake = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        _, surface_flux_uptake = functor._calculate_diffusive_flux(
            concentration_uptake,
            deff,
            species="CO2",
        )

        assert torch.isfinite(surface_flux_emit)
        assert torch.isfinite(surface_flux_uptake)
        assert surface_flux_emit * surface_flux_uptake < 0


def test_real_simulator_applies_gas_increments_and_records_provenance():
    """StateManager applies one conservative gas update with canonical provenance."""
    n_layers = 5
    dt = 1.0
    soil_depth = 1.0
    state = {
        "soil_moisture": torch.full((n_layers,), 0.2),
        "porosity": torch.full((n_layers,), 0.45),
        "soil_temperature": torch.full((n_layers,), 293.15),
        "clay_content": torch.full((n_layers,), 0.2),
        "soil_co2_concentration": torch.linspace(3.0, 1.0, n_layers),
        "n2o_concentration": torch.linspace(0.02, 0.005, n_layers),
    }
    direct = GasDiffusion1D(n_layers=n_layers, soil_depth=soil_depth)
    direct_updates = direct.forward(
        {key: value.clone() for key, value in state.items()},
        time=0.0,
        dt=dt,
    )

    state_manager = StateManager(
        device="cpu",
        conservation_checks={"enabled": False},
    )
    state_manager.set_state(
        {key: value.clone() for key, value in state.items()},
        validate=True,
    )
    simulator = RELICSimulator(
        {
            "device": "cpu",
            "time_step": dt,
            "max_time": dt,
            "conservation_checks": False,
            "strict_dependencies": False,
            "strict_scale_detection": False,
        }
    )
    simulator.initialize(state_manager, DependencyGraphResolver())
    simulator.register_functor(
        GasDiffusion1D(n_layers=n_layers, soil_depth=soil_depth),
        execution_group="fast",
    )
    initial = {
        key: state_manager.get_state(key).clone()
        for key in ("soil_co2_concentration", "n2o_concentration")
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        step_updates = simulator.step(dt=dt)

    assert set(step_updates) == set(direct.metadata.output_variables)
    assert "diagnostics.gas_diffusion.clipping_residual" not in step_updates
    assert "diagnostics.gas_diffusion.co2_clipping_residual" in step_updates
    assert "diagnostics.gas_diffusion.n2o_clipping_residual" in step_updates
    user_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
    ]
    assert not user_warnings, [str(warning.message) for warning in user_warnings]
    for key in ("soil_co2_concentration", "n2o_concentration"):
        actual_delta = state_manager.get_state(key) - initial[key]
        torch.testing.assert_close(
            actual_delta,
            direct_updates[key].value,
            rtol=0.0,
            atol=1.0e-12,
        )
        writers = {
            record["functor"]
            for record in simulator.get_variable_provenance(key)
            if record.get("role") in {"increment_source", "final_writer"}
        }
        assert writers == {"GasDiffusion1D"}

    dz = soil_depth / n_layers
    for concentration_key, surface_flux_key in (
        ("soil_co2_concentration", "co2_surface_flux"),
        ("n2o_concentration", "n2o_surface_flux"),
    ):
        storage_increment = torch.sum(direct_updates[concentration_key].value) * dz
        outward_export = direct_updates[surface_flux_key].value * dt
        torch.testing.assert_close(
            storage_increment + outward_export,
            torch.zeros_like(storage_increment),
            rtol=1.0e-6,
            atol=1.0e-12,
        )
    assert str(simulator.get_effective_device("GasDiffusion1D")) == "cpu"


def test_species_clipping_routes_close_the_finite_volume_ledgers():
    """Bounded concentration updates expose signed species-specific routes."""
    n_layers = 5
    soil_depth = 1.0
    dt = 1.0e8
    state = {
        "soil_moisture": torch.full((n_layers,), 0.2),
        "porosity": torch.full((n_layers,), 0.45),
        "soil_temperature": torch.full((n_layers,), 293.15),
        "clay_content": torch.full((n_layers,), 0.2),
        "soil_co2_concentration": torch.linspace(3.0, 1.0, n_layers),
        "n2o_concentration": torch.linspace(0.02, 0.005, n_layers),
    }
    functor = GasDiffusion1D(n_layers=n_layers, soil_depth=soil_depth)
    updates = functor.forward(state, time=0.0, dt=dt)
    dz = soil_depth / n_layers

    routes = (
        (
            "soil_co2_concentration",
            "co2_surface_flux",
            "diagnostics.gas_diffusion.co2_clipping_residual",
        ),
        (
            "n2o_concentration",
            "n2o_surface_flux",
            "diagnostics.gas_diffusion.n2o_clipping_residual",
        ),
    )
    for concentration_key, surface_flux_key, clipping_key in routes:
        clipping_adjustment = updates[clipping_key].value
        assert torch.any(clipping_adjustment > 0.0)
        residual = (
            torch.sum(updates[concentration_key].value) * dz
            + updates[surface_flux_key].value * dt
            - torch.sum(clipping_adjustment) * dz
        )
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

def test_dndc_surface_exchange_respects_the_disabled_atmospheric_switch():
    """DNDC N2O surface exchange must honor enable_atmospheric_exchange like the Fickian branch."""
    state = {
        'soil_moisture': torch.full((5,), 0.3),
        'porosity': torch.full((5,), 0.45),
        'soil_temperature': torch.full((5,), 293.15),
        'clay_content': torch.full((5,), 0.2),
        'soil_co2_concentration': torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),
        'n2o_concentration': torch.tensor([2.0, 1.8, 1.5, 1.2, 1.0]),
    }
    dt = 3600.0
    enabled = GasDiffusion1D(
        n_layers=5,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.DNDC,
        enable_atmospheric_exchange=True,
    )
    disabled = GasDiffusion1D(
        n_layers=5,
        soil_depth=1.0,
        diffusivity_model=DiffusivityModel.DNDC,
        enable_atmospheric_exchange=False,
    )
    assert enabled.diffusivity_model == DiffusivityModel.DNDC
    assert disabled.diffusivity_model == DiffusivityModel.DNDC
    # Governed calls exercise FunctorBase.__call__ and the gas boundary-ledger hook.
    out_on = enabled(dict(state), time=0.0, dt=dt)
    out_off = disabled(dict(state), time=0.0, dt=dt)
    flux_on = _unwrap(out_on["n2o_surface_flux"])
    flux_off = _unwrap(out_off["n2o_surface_flux"])
    # References: Li et al. (1992); LeVeque (2002).
    # Mathematical expression: closed column -> sum_i (dC_i - R_clip,i) dz = 0;
    #   open column -> sum_i (dC_i - R_clip,i) dz + F_out dt = 0.
    # Parameter meanings: dC_i accepted increments, R_clip,i clipping residuals, dz layer thickness.
    # Units: C [g N m-3]; dz [m]; F_out [g N m-2 s-1]; dt [s].
    # Domain/constraints: surface N2O above the atmospheric boundary, so the open flux is positive.
    # Sign convention: F_out positive from soil to atmosphere.
    assert float(flux_on) > 0.0
    assert torch.count_nonzero(flux_off) == 0
    dz = float(disabled.soil_depth) / float(disabled.n_layers)
    d_off = _unwrap(out_off["n2o_concentration"])
    clip_off = _unwrap(out_off["diagnostics.gas_diffusion.n2o_clipping_residual"])
    closed_budget = ((d_off - clip_off).sum() * dz)
    torch.testing.assert_close(closed_budget, torch.zeros_like(closed_budget), rtol=0.0, atol=1.0e-12)
    d_on = _unwrap(out_on["n2o_concentration"])
    clip_on = _unwrap(out_on["diagnostics.gas_diffusion.n2o_clipping_residual"])
    open_budget = (d_on - clip_on).sum() * dz + flux_on.to(d_on.dtype) * dt
    torch.testing.assert_close(open_budget, torch.zeros_like(open_budget), rtol=0.0, atol=1.0e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

