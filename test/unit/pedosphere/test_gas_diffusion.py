"""
Unit tests for GasDiffusion1D functor.

These tests verify the correctness of gas diffusion implementation
including effective diffusivity models, temperature effects, and boundary conditions.
"""

import inspect

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.pedosphere.transport.gas_diffusion import (
    GasDiffusion1D,
    GasSpecies,
    DiffusivityModel
)
from relic.parameters import get_global_manager


pytestmark = pytest.mark.usefixtures(
    "pedosphere_transport_constructor_overrides"
)


def _unwrap(value):
    """Return the tensor carried by a canonical state update."""
    return value.value if isinstance(value, UpdateItem) else value


class TestGasDiffusion1D:
    """Test suite for GasDiffusion1D functor."""
    
    @pytest.fixture
    def functor(self):
        """Create a basic GasDiffusion1D functor for testing."""
        return GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK,
            enable_temperature_correction=True
        )
    
    @pytest.fixture
    def basic_state(self):
        """Create a basic state dictionary for testing."""
        return {
            'soil_moisture': torch.tensor([0.2, 0.18, 0.15, 0.12, 0.1]),  # WFPS
            'porosity': torch.tensor([0.45, 0.42, 0.4, 0.38, 0.35]),
            'soil_temperature': torch.tensor([288.15, 287.15, 286.15, 285.15, 284.15]),  # K
            'clay_content': torch.tensor([0.15, 0.18, 0.2, 0.22, 0.25]),
            'soil_co2_concentration': torch.tensor([2.0, 1.8, 1.5, 1.2, 1.0]),  # g C m-3
            'n2o_concentration': torch.tensor([0.002, 0.0018, 0.0015, 0.0012, 0.001])  # g N m-3
        }
    
    def test_initialization(self):
        """Test proper initialization of the functor."""
        functor = GasDiffusion1D()
        pm = get_global_manager(preload_all=True, warmup_cache=False)
        
        # Check basic properties
        expected_layers = int(pm.get("pedosphere.grid.n_layers"))
        expected_depth = float(pm.get("pedosphere.grid.soil_depth"))
        expected_model = pm.get("pedosphere.transport.diffusion_model")
        assert functor.n_layers == expected_layers
        assert float(functor.soil_depth) == expected_depth
        assert functor.diffusivity_model == DiffusivityModel(expected_model)
        
        # Check metadata
        assert functor.metadata.name == "GasDiffusion1D"
        from relic.core.functor import ProcessCategory
        assert functor.metadata.category == ProcessCategory.TRANSPORT_MECHANISM
        assert "Canonical IPHS:" in functor.metadata.mathematical_expression
        assert "literature formulas are stored only" in functor.metadata.mathematical_expression
        
        # Check species configuration
        assert "CO2" in functor.species
        assert "N2O" in functor.species
        expected_co2_d0 = float(pm.get("pedosphere.physical.co2_diffusivity_air_stp"))
        expected_n2o_d0 = float(pm.get("pedosphere.physical.n2o_diffusivity_air_stp"))
        assert functor.species["CO2"].d0_at_stp == expected_co2_d0
        assert functor.species["N2O"].d0_at_stp == expected_n2o_d0
    
    def test_diffusivity_models(self):
        """Test different diffusivity model options."""
        # Millington-Quirk model
        mq_functor = GasDiffusion1D(diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK)
        assert mq_functor.diffusivity_model == DiffusivityModel.MILLINGTON_QUIRK
        
        # Penman model
        penman_functor = GasDiffusion1D(diffusivity_model=DiffusivityModel.PENMAN)
        assert penman_functor.diffusivity_model == DiffusivityModel.PENMAN
        
        # DNDC model
        dndc_functor = GasDiffusion1D(diffusivity_model=DiffusivityModel.DNDC)
        assert dndc_functor.diffusivity_model == DiffusivityModel.DNDC
    
    def test_species_configuration(self):
        """Test species parameters are loaded from the parameter manager."""
        functor = GasDiffusion1D()
        pm = get_global_manager(preload_all=True, warmup_cache=False)
        expected_exp = float(pm.get("pedosphere.physical.gas_diffusivity_temperature_exponent"))
        assert functor.species["CO2"].temperature_exponent == expected_exp
        assert functor.species["N2O"].temperature_exponent == expected_exp
    
    def test_get_dependencies(self, functor):
        """Test dependency declaration."""
        dependencies = functor.get_dependencies()
        if isinstance(dependencies, dict):
            deps_list = dependencies.get("required", []) + dependencies.get("optional", [])
        else:
            deps_list = dependencies

        assert isinstance(deps_list, list)
        assert any(
            name in deps_list
            for name in [
                "first_order_decomposition",
                "microbial_enzyme_decomposition",
                "nitrification",
                "denitrification"
            ]
        )
    
    def test_get_required_inputs(self, functor):
        """Test required inputs declaration."""
        inputs = functor.get_required_inputs()
        
        assert inputs == [
            "soil_moisture",
            "porosity",
            "soil_temperature",
            "clay_content",
            "soil_co2_concentration",
            "n2o_concentration",
        ]
    
    def test_get_output_specifications(self, functor):
        """Test output specifications."""
        outputs = functor.get_output_specifications()
        
        # Check required outputs
        assert "heterotrophic_respiration" not in outputs
        assert "n2o_flux" not in outputs
        assert "co2_surface_flux" in outputs
        assert "n2o_surface_flux" in outputs

        # Check units
        assert outputs["co2_surface_flux"].units == "g C m-2 s-1"
        assert outputs["n2o_surface_flux"].units == "g N m-2 s-1"
    
    def test_input_validation(self, functor, basic_state):
        """Test input validation."""
        # Valid state should pass
        errors = functor.validate_inputs(basic_state)
        assert len(errors) == 0
        
        # Missing required input
        invalid_state = basic_state.copy()
        del invalid_state['porosity']
        errors = functor.validate_inputs(invalid_state)
        assert len(errors) > 0
        assert any("Missing required input" in error for error in errors)
        
        # Physical constraint violation - porosity > 1
        invalid_state = basic_state.copy()
        invalid_state['porosity'] = torch.tensor([1.2, 0.4, 0.3, 0.2, 0.1])
        errors = functor.validate_inputs(invalid_state)
        assert len(errors) > 0
        assert any("must be in [0,1]" in error for error in errors)
        
        # Temperature out of range
        invalid_state = basic_state.copy()
        invalid_state['soil_temperature'] = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])  # Too low/high
        errors = functor.validate_inputs(invalid_state)
        assert len(errors) > 0
        assert any("soil_temperature out of range" in error for error in errors)
        
        # Negative concentration
        invalid_state = basic_state.copy()
        invalid_state['soil_co2_concentration'] = torch.tensor([2.0, -1.0, 1.5, 1.2, 1.0])
        errors = functor.validate_inputs(invalid_state)
        assert len(errors) > 0
        assert any("must be non-negative" in error for error in errors)
    
    def test_effective_diffusivity_millington_quirk(self, functor, basic_state):
        """Test Millington-Quirk diffusivity model."""
        functor.diffusivity_model = DiffusivityModel.MILLINGTON_QUIRK
        
        theta = functor._align_profile(basic_state["soil_moisture"])
        porosity = functor._align_profile(basic_state["porosity"])
        temperature = functor._align_profile(basic_state["soil_temperature"])
        clay = functor._align_profile(basic_state["clay_content"])
        
        # Calculate air-filled porosity
        air_porosity = torch.clamp(porosity - theta, min=1e-6)
        
        # Calculate diffusivity
        deff_co2 = functor._calculate_effective_diffusivity(
            air_porosity, porosity, temperature, clay, "CO2"
        )
        
        # All values should be positive
        assert torch.all(deff_co2 > 0)
        
        # Should be less than free-air diffusivity
        d0_co2 = functor.species["CO2"].d0_at_stp
        assert torch.all(deff_co2 < d0_co2)
        
        # Higher air porosity should give higher diffusivity
        high_air_porosity = air_porosity * 2
        deff_high = functor._calculate_effective_diffusivity(
            high_air_porosity, porosity, temperature, clay, "CO2"
        )
        assert torch.all(deff_high > deff_co2)
    
    def test_effective_diffusivity_penman(self, functor, basic_state):
        """Test Penman diffusivity model."""
        functor.diffusivity_model = DiffusivityModel.PENMAN
        
        theta = functor._align_profile(basic_state["soil_moisture"])
        porosity = functor._align_profile(basic_state["porosity"])
        temperature = functor._align_profile(basic_state["soil_temperature"])
        clay = functor._align_profile(basic_state["clay_content"])
        
        air_porosity = torch.clamp(porosity - theta, min=1e-6)
        
        deff_co2 = functor._calculate_effective_diffusivity(
            air_porosity, porosity, temperature, clay, "CO2"
        )
        
        # All values should be positive
        assert torch.all(deff_co2 > 0)
        
        # Penman model: Deff = 0.66 * D0 * theta_a
        d0_co2 = functor.species["CO2"].d0_at_stp
        expected_approx = 0.66 * d0_co2 * air_porosity
        
        # Should be in the right order of magnitude
        assert torch.all(deff_co2 / expected_approx > 0.1)
        assert torch.all(deff_co2 / expected_approx < 10.0)
    
    def test_effective_diffusivity_dndc(self, functor, basic_state):
        """DNDC uses one dimensionless coefficient set and one formula path."""
        functor.diffusivity_model = DiffusivityModel.DNDC

        pm = get_global_manager(preload_all=True, warmup_cache=False)
        parameter_keys = set(functor.required_parameters())
        assert "pedosphere.transport.n2o_dndc_base_coefficient" in parameter_keys
        assert "pedosphere.transport.n2o_dndc_base_rate" not in parameter_keys
        assert (
            "pedosphere.transport.n2o_dndc_porosity_base_coefficient"
            not in parameter_keys
        )
        coefficient_keys = {
            "pedosphere.transport.n2o_dndc_base_coefficient": 0.018,
            "pedosphere.transport.n2o_dndc_clay_effect": 0.124,
            "pedosphere.transport.n2o_dndc_porosity_effect": 0.013,
            "pedosphere.transport.n2o_dndc_clay_porosity_coefficient": 0.016,
        }
        for key, expected_value in coefficient_keys.items():
            spec = pm.get_parameter(key)
            assert spec is not None
            assert spec.units == "dimensionless"
            assert float(pm.get(key)) == pytest.approx(expected_value)

        effective_source = inspect.getsource(
            GasDiffusion1D._calculate_effective_diffusivity
        )
        surface_source = inspect.getsource(GasDiffusion1D._calculate_dndc_n2o_flux)
        assert "_dndc_n2o_diffusion_factor" in effective_source
        assert "_dndc_n2o_diffusion_factor" in surface_source
        
        theta = functor._align_profile(basic_state["soil_moisture"])
        porosity = functor._align_profile(basic_state["porosity"])
        temperature = functor._align_profile(basic_state["soil_temperature"])
        clay = functor._align_profile(basic_state["clay_content"])
        
        air_porosity = torch.clamp(porosity - theta, min=1e-6)
        
        # DNDC model is specific to N2O
        deff_n2o = functor._calculate_effective_diffusivity(
            air_porosity, porosity, temperature, clay, "N2O"
        )
        
        wfps = torch.clamp(theta / porosity, 0.0, 1.0)
        temp_c = temperature - functor.celsius_offset
        expected_factor = (1.0 - wfps) * (
            0.018
            + 0.124 * clay
            + (0.013 - 0.016 * clay) * torch.pow(2.0, temp_c / 20.0) * porosity
        )
        layer_depth = float(functor.soil_depth) / float(functor.n_layers)
        expected_deff = (
            expected_factor * layer_depth**2 / functor.seconds_per_hour
        )
        assert torch.allclose(deff_n2o, expected_deff, rtol=1.0e-6, atol=1.0e-12)
        
        # CO2 should fall back to Millington-Quirk
        deff_co2 = functor._calculate_effective_diffusivity(
            air_porosity, porosity, temperature, clay, "CO2"
        )
        assert torch.all(deff_co2 > 0)
    
    def test_temperature_correction(self, functor, basic_state):
        """Test temperature correction for diffusivity."""
        theta = functor._align_profile(basic_state["soil_moisture"])
        porosity = functor._align_profile(basic_state["porosity"])
        clay = functor._align_profile(basic_state["clay_content"])
        
        air_porosity = torch.clamp(porosity - theta, min=1e-6)
        
        # Reference temperature
        temp_ref = torch.tensor([293.15] * 5)  # 20 C
        
        # Higher temperature
        temp_high = torch.tensor([303.15] * 5)  # 30 C
        
        # Calculate diffusivity at both temperatures
        deff_ref = functor._calculate_effective_diffusivity(
            air_porosity, porosity, temp_ref, clay, "CO2"
        )
        deff_high = functor._calculate_effective_diffusivity(
            air_porosity, porosity, temp_high, clay, "CO2"
        )
        
        # Higher temperature should give higher diffusivity
        assert torch.all(deff_high > deff_ref)
        
        # Test without temperature correction
        functor.enable_temperature_correction = False
        deff_no_temp = functor._calculate_effective_diffusivity(
            air_porosity, porosity, temp_high, clay, "CO2"
        )
        
        # Should be same as reference when temperature correction is disabled
        deff_ref_no_temp = functor._calculate_effective_diffusivity(
            air_porosity, porosity, temp_ref, clay, "CO2"
        )
        assert torch.allclose(deff_no_temp, deff_ref_no_temp)
    
    def test_diffusive_flux_calculation(self, functor, basic_state):
        """Test diffusive flux calculation using central differences."""
        theta = functor._align_profile(basic_state["soil_moisture"])
        porosity = functor._align_profile(basic_state["porosity"])
        temperature = functor._align_profile(basic_state["soil_temperature"])
        clay = functor._align_profile(basic_state["clay_content"])
        co2_conc = functor._align_profile(basic_state["soil_co2_concentration"])
        
        air_porosity = torch.clamp(porosity - theta, min=1e-6)
        
        # Calculate effective diffusivity
        deff = functor._calculate_effective_diffusivity(
            air_porosity, porosity, temperature, clay, "CO2"
        )
        
        # Calculate flux profile
        flux_profile, surface_flux = functor._calculate_diffusive_flux(
            concentration=co2_conc,
            deff=deff,
            species="CO2"
        )
        
        # Flux profile should have n_layers-1 elements
        assert flux_profile.shape[0] == functor.n_layers - 1
        
        # Surface flux should be finite and non-zero for this non-uniform profile.
        assert torch.isfinite(surface_flux)
        assert torch.abs(surface_flux) > 0
    
    def test_dndc_n2o_flux_calculation(self, functor, basic_state):
        """DNDC surface exchange is dimensional and positive out of soil."""
        functor.diffusivity_model = DiffusivityModel.DNDC
        
        theta = functor._align_profile(basic_state["soil_moisture"])
        clay = functor._align_profile(basic_state["clay_content"])
        porosity = functor._align_profile(basic_state["porosity"])
        temperature = functor._align_profile(basic_state["soil_temperature"])
        n2o_conc = functor._align_profile(basic_state["n2o_concentration"])
        
        # Calculate DNDC N2O flux
        n2o_flux = functor._calculate_dndc_n2o_flux(
            n2o_conc, theta, clay, porosity, temperature
        )
        
        wfps = torch.clamp(
            theta / porosity,
            float(
                functor.get_parameter_value(
                    "wfps_minimum", namespace="pedosphere.numerical", required=True
                )
            ),
            float(
                functor.get_parameter_value(
                    "wfps_maximum", namespace="pedosphere.numerical", required=True
                )
            ),
        )
        temp_c = temperature - functor.celsius_offset
        diffusion_factor = (1.0 - wfps) * (
            0.018
            + 0.124 * clay
            + (0.013 - 0.016 * clay)
            * torch.pow(2.0, temp_c / 20.0)
            * porosity
        )
        layer_depth = float(functor.soil_depth) / float(functor.n_layers)
        atmospheric_n2o = torch.as_tensor(
            functor.species["N2O"].atmospheric_concentration,
            dtype=n2o_conc.dtype,
            device=n2o_conc.device,
        )
        expected_flux = (
            (n2o_conc[0] - atmospheric_n2o)
            * layer_depth
            * diffusion_factor[0]
            / functor.seconds_per_hour
        )
        assert torch.allclose(n2o_flux, expected_flux, rtol=1.0e-6, atol=1.0e-12)
        assert n2o_flux > 0.0
        
        # Should depend on surface concentration
        high_conc = n2o_conc.clone()
        high_conc[0] = high_conc[0] * 2
        
        high_flux = functor._calculate_dndc_n2o_flux(
            high_conc, theta, clay, porosity, temperature
        )
        
        assert high_flux > n2o_flux

        low_conc = n2o_conc.clone()
        low_conc[0] = atmospheric_n2o * 0.5
        uptake_flux = functor._calculate_dndc_n2o_flux(
            low_conc, theta, clay, porosity, temperature
        )
        assert uptake_flux < 0.0
    
    def test_forward_execution(self, functor, basic_state):
        """Test complete forward execution."""
        dt = 300.0  # 5 minutes
        time = 0.0
        
        # Execute forward pass
        output = functor.forward(basic_state, time, dt)
        
        # Check output structure
        assert isinstance(output, dict)
        
        # Check required outputs exist
        assert "heterotrophic_respiration" not in output
        assert "n2o_flux" not in output
        assert "co2_surface_flux" in output
        assert "n2o_surface_flux" in output

        # Check output properties
        # Surface/diagnostic fluxes should be finite; sign is controlled by gradient direction.
        assert torch.all(torch.isfinite(_unwrap(output["co2_surface_flux"])))
        assert torch.all(torch.isfinite(_unwrap(output["n2o_surface_flux"])))
    
    def test_atmospheric_boundary_condition(self, functor, basic_state):
        """Test atmospheric exchange boundary condition."""
        # With atmospheric exchange enabled
        functor.enable_atmospheric_exchange = True
        
        output_with_atm = functor.forward(basic_state, 0.0, 300.0)
        
        # Without atmospheric exchange
        functor.enable_atmospheric_exchange = False
        
        output_no_atm = functor.forward(basic_state, 0.0, 300.0)
        
        # Surface fluxes should be different
        co2_flux_with_atm = _unwrap(output_with_atm["co2_surface_flux"])
        co2_flux_no_atm = _unwrap(output_no_atm["co2_surface_flux"])
        
        # With atmospheric exchange should generally give higher fluxes
        # (due to concentration gradient with atmosphere)
        assert co2_flux_with_atm != co2_flux_no_atm
    
    def test_model_comparison(self, basic_state):
        """Test different diffusivity models give different results."""
        dt = 300.0
        time = 0.0
        
        # Create functors with different models
        mq_functor = GasDiffusion1D(
            n_layers=5,
            diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK,
        )
        penman_functor = GasDiffusion1D(
            n_layers=5,
            diffusivity_model=DiffusivityModel.PENMAN,
        )
        dndc_functor = GasDiffusion1D(
            n_layers=5,
            diffusivity_model=DiffusivityModel.DNDC,
        )
        
        # Execute all models
        mq_output = mq_functor.forward(basic_state, time, dt)
        penman_output = penman_functor.forward(basic_state, time, dt)
        dndc_output = dndc_functor.forward(basic_state, time, dt)
        
        # Results should be different between models
        # Use more strict tolerance for small values
        assert not torch.allclose(_unwrap(mq_output["co2_surface_flux"]), _unwrap(penman_output["co2_surface_flux"]), rtol=1e-3, atol=1e-10)
        assert not torch.allclose(_unwrap(mq_output["n2o_surface_flux"]), _unwrap(dndc_output["n2o_surface_flux"]), rtol=1e-3, atol=1e-10)
        
        # All should be finite; sign is physically directional.
        for output in [mq_output, penman_output, dndc_output]:
            assert torch.all(torch.isfinite(_unwrap(output["co2_surface_flux"])))
            assert torch.all(torch.isfinite(_unwrap(output["n2o_surface_flux"])))
    
    def test_moisture_effects(self, functor, basic_state):
        """Test effects of soil moisture on gas diffusion."""
        # Dry soil (low water content)
        dry_state = basic_state.copy()
        dry_state['soil_moisture'] = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05])
        
        # Wet soil (high water content)
        wet_state = basic_state.copy()
        wet_state['soil_moisture'] = torch.tensor([0.35, 0.35, 0.35, 0.35, 0.35])
        
        dt = 300.0
        time = 0.0
        
        # Execute both scenarios
        dry_output = functor.forward(dry_state, time, dt)
        wet_output = functor.forward(wet_state, time, dt)
        
        # Dry soil should have higher gas-flux magnitude (more air-filled porosity).
        assert torch.abs(_unwrap(dry_output["co2_surface_flux"])) > torch.abs(_unwrap(wet_output["co2_surface_flux"]))
        assert torch.abs(_unwrap(dry_output["n2o_surface_flux"])) > torch.abs(_unwrap(wet_output["n2o_surface_flux"]))
    
    def test_concentration_gradient_effects(self, functor, basic_state):
        """Test effects of concentration gradients on fluxes."""
        # High gradient case (high surface concentration)
        high_grad_state = basic_state.copy()
        high_grad_state['soil_co2_concentration'] = torch.tensor([5.0, 1.0, 0.5, 0.2, 0.1])
        high_grad_state['n2o_concentration'] = torch.tensor([0.01, 0.001, 0.0005, 0.0002, 0.0001])
        
        # Low gradient case (uniform concentration)
        low_grad_state = basic_state.copy()
        low_grad_state['soil_co2_concentration'] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        low_grad_state['n2o_concentration'] = torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001])
        
        dt = 300.0
        time = 0.0
        
        # Execute both scenarios
        high_grad_output = functor.forward(high_grad_state, time, dt)
        low_grad_output = functor.forward(low_grad_state, time, dt)
        
        # High gradient should produce higher surface-flux magnitude.
        assert torch.abs(_unwrap(high_grad_output["co2_surface_flux"])) > torch.abs(_unwrap(low_grad_output["co2_surface_flux"]))
        assert torch.abs(_unwrap(high_grad_output["n2o_surface_flux"])) > torch.abs(_unwrap(low_grad_output["n2o_surface_flux"]))

    def test_diffusion_delta_preserves_sign_and_state_non_negative(self, functor, basic_state):
        """Diffusive delta must allow negative increments while keeping updated state non-negative."""
        state = basic_state.copy()
        state["soil_co2_concentration"] = torch.tensor([5.0, 2.0, 1.0, 0.6, 0.3])
        state["n2o_concentration"] = torch.tensor([0.01, 0.004, 0.002, 0.001, 0.0005])

        output = functor.forward(state, time=0.0, dt=300.0)

        delta_co2 = _unwrap(output["soil_co2_concentration"])
        delta_n2o = _unwrap(output["n2o_concentration"])
        assert output["soil_co2_concentration"].mode == UpdateMode.ACCUMULATE
        assert output["soil_co2_concentration"].per_timestep is True
        assert output["n2o_concentration"].mode == UpdateMode.ACCUMULATE
        assert output["n2o_concentration"].per_timestep is True
        assert torch.any(delta_co2 < 0.0)
        assert torch.any(delta_n2o < 0.0)

        co2_new = state["soil_co2_concentration"] + delta_co2
        n2o_new = state["n2o_concentration"] + delta_n2o
        assert torch.all(co2_new >= 0.0)
        assert torch.all(n2o_new >= 0.0)

    def test_finite_volume_ledger_has_no_generic_pool_flux_limiter(self):
        """Gas transport uses its species-specific finite-volume ledger only."""
        source = inspect.getsource(GasDiffusion1D.forward)
        assert "stability_controller" not in source
        assert ".limit_fluxes(" not in source
    
    @pytest.mark.parametrize("model", [
        DiffusivityModel.MILLINGTON_QUIRK,
        DiffusivityModel.PENMAN,
        DiffusivityModel.DNDC
    ])
    def test_all_diffusivity_models(self, basic_state, model):
        """Test all diffusivity models work correctly."""
        functor = GasDiffusion1D(n_layers=5, diffusivity_model=model)
        
        dt = 300.0
        time = 0.0
        
        # Should work without errors
        output = functor.forward(basic_state, time, dt)
        
        # Basic checks
        assert torch.all(torch.isfinite(_unwrap(output["co2_surface_flux"])))
        assert torch.all(torch.isfinite(_unwrap(output["n2o_surface_flux"])))


class TestGasSpecies:
    """Test suite for GasSpecies configuration."""
    
    def test_species_creation(self):
        """Test GasSpecies creation and properties."""
        species = GasSpecies(
            name="TEST",
            d0_at_stp=2.0e-5,
            temperature_exponent=1.6,
            atmospheric_concentration=0.5
        )
        
        assert species.name == "TEST"
        assert species.d0_at_stp == 2.0e-5
        assert species.temperature_exponent == 1.6
        assert species.atmospheric_concentration == 0.5
    
    def test_default_values(self):
        """Test optional parameter defaults."""
        species = GasSpecies(
            name="DEFAULT",
            d0_at_stp=1.5e-5,
            temperature_exponent=1.75
        )
        
        assert species.temperature_exponent == 1.75
        assert species.atmospheric_concentration is None  # Not required


class TestDiffusivityModel:
    """Test suite for DiffusivityModel enum."""
    
    def test_enum_values(self):
        """Test DiffusivityModel enum values."""
        assert DiffusivityModel.MILLINGTON_QUIRK.value == "millington_quirk"
        assert DiffusivityModel.PENMAN.value == "penman"
        assert DiffusivityModel.DNDC.value == "dndc"
    
    def test_enum_comparison(self):
        """Test enum comparison."""
        model1 = DiffusivityModel.MILLINGTON_QUIRK
        model2 = DiffusivityModel.MILLINGTON_QUIRK
        model3 = DiffusivityModel.PENMAN
        
        assert model1 == model2
        assert model1 != model3


# Helper functions for testing
def calculate_theoretical_flux(concentration_gradient, diffusivity):
    """
    Calculate theoretical flux using Fick's law.
    J = -D * dC/dz
    """
    return -diffusivity * concentration_gradient


def test_fick_law_consistency():
    """Test that flux calculations are consistent with Fick's law."""
    concentration_gradient = torch.tensor([-2.0, 0.0, 3.0])
    diffusivity = torch.tensor([1.0e-6, 2.0e-6, 4.0e-6])

    flux = calculate_theoretical_flux(concentration_gradient, diffusivity)

    assert torch.allclose(flux, torch.tensor([2.0e-6, 0.0, -1.2e-5]))
    assert torch.all(flux * concentration_gradient <= 0.0)


def test_mass_balance_gas_phase():
    """Test gas phase mass balance."""
    n_layers = 5
    soil_depth = 1.0
    dt = 1.0
    functor = GasDiffusion1D(
        n_layers=n_layers,
        soil_depth=soil_depth,
        enable_atmospheric_exchange=True,
    )
    state = {
        "soil_moisture": torch.full((n_layers,), 0.2),
        "porosity": torch.full((n_layers,), 0.45),
        "soil_temperature": torch.full((n_layers,), 293.15),
        "clay_content": torch.full((n_layers,), 0.2),
        "soil_co2_concentration": torch.linspace(3.0, 1.0, n_layers),
        "n2o_concentration": torch.linspace(0.02, 0.005, n_layers),
    }

    output = functor.forward(state, time=0.0, dt=dt)
    dz = soil_depth / n_layers
    for concentration_key, surface_flux_key in (
        ("soil_co2_concentration", "co2_surface_flux"),
        ("n2o_concentration", "n2o_surface_flux"),
    ):
        storage_increment = torch.sum(_unwrap(output[concentration_key])) * dz
        outward_boundary_export = _unwrap(output[surface_flux_key]) * dt
        assert torch.allclose(
            storage_increment + outward_boundary_export,
            torch.zeros_like(storage_increment),
            rtol=1.0e-6,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

