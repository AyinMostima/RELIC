"""
Comprehensive L1 Unit Tests for Canopy Multi-layer Radiation Transfer

This test suite provides exhaustive testing of the canopy radiation transfer
functionality in the RadiationFunctor class. Tests cover Beer-Lambert multi-layer
calculations, PAR/PPFD derivation, conservation laws, and edge cases.

Test Categories:
1. Configuration and Parameter Tests
2. Direct-Diffuse Separation Tests
3. LAI Profile Generation Tests
4. Beer-Lambert Multi-layer Tests
5. Conservation Law Tests
6. Edge Cases and Boundary Tests
7. Integration and consistency Tests

References for validation values:
- Monsi, M. & Saeki, T. (1953). Light factor and leaf arrangement
- Charles-Edwards, D.A. (1982). Physiological determinants of crop growth  
- Campbell & Norman (1998). Environmental Biophysics
- Erbs et al. (1982). Estimation of the diffuse radiation fraction for hourly,
  daily and monthly-average global radiation
"""

import pytest
import torch

from relic.core.state import UpdateItem, UpdateMode
from relic.parameters import get_global_manager
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.core.feature_flags import feature_manager


def _value(output):
    """Return the tensor payload from explicit forward updates."""
    return output.value if isinstance(output, UpdateItem) else output


def _expected_par_energy(
    radiation: RadiationFunctor,
    shortwave_down: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the declared PAR-energy partition for conservation assertions."""
    return shortwave_down * radiation._get_canopy_parameter('par_fraction_of_sw')


def _extraterrestrial_horizontal(
    shortwave_down: torch.Tensor,
    clearness_index: float = 0.7,
) -> torch.Tensor:
    """Return a consistent extraterrestrial-horizontal flux for helper tests."""
    return shortwave_down / clearness_index


def _radiation_with_overrides(monkeypatch, **overrides):
    """Build RadiationFunctor with calibrated ParameterManager values."""
    manager = get_global_manager()
    original_get = manager.get
    full_overrides = {
        f'atmosphere.radiation.canopy.{key}': value
        for key, value in overrides.items()
    }
    resolved_overrides = {}
    for key, value in full_overrides.items():
        spec = manager.get_parameter(key)
        assert spec is not None
        if spec.requires_grad:
            parameter = torch.nn.Parameter(
                torch.as_tensor(value, dtype=torch.float32),
                requires_grad=True,
            )
            monkeypatch.setitem(manager.nn_parameters, key, parameter)
            resolved_overrides[key] = parameter
        else:
            resolved_overrides[key] = value

    def _get(key, *args, **kwargs):
        if key in resolved_overrides:
            return resolved_overrides[key]
        return original_get(key, *args, **kwargs)

    monkeypatch.setattr(manager, 'get', _get)
    return RadiationFunctor(parameter_manager=manager)


@pytest.fixture(autouse=True)
def _selected_canopy_topology():
    """Select canopy topology for canopy-specific tests and restore it after use."""
    was_enabled = feature_manager.is_enabled('canopy_layers_enabled')
    feature_manager.enable('canopy_layers_enabled')
    try:
        yield
    finally:
        if was_enabled:
            feature_manager.enable('canopy_layers_enabled')
        else:
            feature_manager.disable('canopy_layers_enabled')


class TestCanopyRadiationConfiguration:
    """Test canopy radiation configuration and parameter handling."""
    
    @pytest.fixture
    def device(self):
        """Test device (CPU or CUDA if available)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_canopy_disabled_by_default(self, device):
        """Test that canopy radiation is disabled by default."""
        was_enabled = feature_manager.is_enabled('canopy_layers_enabled')
        feature_manager.disable('canopy_layers_enabled')
        radiation = RadiationFunctor(device=device)
        try:
            assert not feature_manager.is_enabled('canopy_layers_enabled')
            assert 'leaf_area_index' not in radiation.metadata.input_variables
            
            # Test forward with canopy disabled
            state = {
                'air_temperature': torch.tensor(298.15),
                'incoming_shortwave_potential': torch.tensor(1200.0),
                'cos_zenith': torch.tensor(0.8),
                'surface_temperature': torch.tensor(295.15),
                'relative_humidity': torch.tensor(0.7),
                'surface_albedo': torch.tensor(0.2),
                'surface_emissivity': torch.tensor(0.97),
                'cloud_fraction': torch.tensor(0.0),
            }
            
            result = radiation.forward(state, time=43200.0, dt=60.0)
            
            # PPFD is canonical for all runs; only layer outputs are conditional.
            assert 'ppfd' in result
            assert torch.isfinite(_value(result['ppfd'])).all()
            assert 'par' not in result
            canopy_keys = ['fpar', 'apar_layer']
            for key in canopy_keys:
                assert key not in result, f"Canopy output '{key}' found when disabled"
        finally:
            if was_enabled:
                feature_manager.enable('canopy_layers_enabled')

    def test_canopy_selection_cannot_change_after_initialization(self, device):
        """Runtime feature changes must not invalidate the registered IO graph."""
        was_enabled = feature_manager.is_enabled('canopy_layers_enabled')
        feature_manager.disable('canopy_layers_enabled')
        radiation = RadiationFunctor(device=device)
        feature_manager.enable('canopy_layers_enabled')
        try:
            with pytest.raises(RuntimeError, match='changed after initialization'):
                radiation._compute_canopy_outputs(
                    {},
                    torch.tensor(100.0),
                    torch.tensor(0.0),
                    False,
                    {},
                )
        finally:
            if not was_enabled:
                feature_manager.disable('canopy_layers_enabled')
    
    def test_canopy_enabled_configuration(self, device):
        """The feature flag selects canopy topology and manager parameters."""
        radiation = RadiationFunctor(device=device)

        assert feature_manager.is_enabled('canopy_layers_enabled')
        assert radiation._get_canopy_parameter('model_type') == 'beer_lambert'
        assert radiation._get_canopy_parameter('layers') >= 1
        assert radiation._get_canopy_parameter('k_b') > 0.0
        assert radiation._get_canopy_parameter('k_d') > 0.0
        assert 'leaf_area_index' in radiation.metadata.input_variables
    
    def test_parameter_bounds_validation(self, device):
        """Test parameter bounds are respected."""
        radiation = RadiationFunctor(device=device)
        
        # Check parameters are within bounds
        layers = radiation._get_canopy_parameter('layers')
        k_b = radiation._get_canopy_parameter('k_b') 
        k_d = radiation._get_canopy_parameter('k_d')
        
        assert 1 <= layers <= 100
        assert 0.2 <= k_b <= 1.2
        assert 0.3 <= k_d <= 1.5


class TestDirectDiffuseSeparation:
    """Test direct-diffuse PAR separation and PPFD calculation."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.radiation = RadiationFunctor()
    
    def test_par_calculation_from_shortwave(self):
        """Test PAR calculation from shortwave radiation."""
        shortwave_down = torch.tensor(800.0)  # W m^-2
        
        components = self.radiation._split_direct_diffuse(
            shortwave_down,
            extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
        )
        
        # Check PAR calculation
        expected_par = 800.0 * 0.45  # 360 W m^-2
        assert torch.isclose(components['par_total'], torch.tensor(expected_par), rtol=0.01)
        
        # Check PPFD calculation  
        expected_ppfd = expected_par * 4.6  # Approximately 1656 umol m^-2 s^-1
        actual_ppfd = components['ppfd_dir'] + components['ppfd_dif']
        assert torch.isclose(actual_ppfd, torch.tensor(expected_ppfd), rtol=0.01)
    
    @pytest.mark.parametrize(
        ('clearness_index', 'expected_diffuse_fraction'),
        [
            (0.1, 0.991),
            (0.5, 0.65915),
            (0.9, 0.165),
        ],
    )
    def test_erbs_hourly_diffuse_fraction_branches(
        self,
        clearness_index,
        expected_diffuse_fraction,
    ):
        """Evaluate all three published Erbs hourly-correlation branches."""
        extraterrestrial_horizontal = torch.tensor(1000.0)
        shortwave_down = extraterrestrial_horizontal * clearness_index

        components = self.radiation._split_direct_diffuse(
            shortwave_down,
            extraterrestrial_horizontal=extraterrestrial_horizontal,
        )

        assert torch.isclose(
            components['f_dif'],
            torch.tensor(expected_diffuse_fraction),
            rtol=1e-6,
            atol=1e-7,
        )
    
    def test_global_radiation_cannot_exceed_extraterrestrial_horizontal(self):
        """Reject a clearness index above the Erbs correlation domain."""
        with pytest.raises(ValueError, match='exceeds extraterrestrial horizontal'):
            self.radiation._split_direct_diffuse(
                torch.tensor(1001.0),
                extraterrestrial_horizontal=torch.tensor(1000.0),
            )
    
    def test_conservation_direct_diffuse_split(self):
        """Test conservation of PAR in direct-diffuse split."""
        shortwave_down = torch.tensor(1000.0)

        components = self.radiation._split_direct_diffuse(
            shortwave_down,
            extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
        )
        
        par_total = components['par_total']
        par_dir = components['par_dir']
        par_dif = components['par_dif']
        
        # Check conservation
        par_sum = par_dir + par_dif
        assert torch.isclose(par_total, par_sum, rtol=1e-6), \
            f"PAR conservation failed: {par_total} != {par_sum}"
    
    def test_ppfd_calculation_consistency(self):
        """Test PPFD calculation consistency."""
        shortwave_down = torch.tensor(750.0)
        
        components = self.radiation._split_direct_diffuse(
            shortwave_down,
            extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
        )
        
        # Check PPFD components sum to total
        ppfd_total_calc = components['ppfd_dir'] + components['ppfd_dif']
        par_total = components['par_total']
        ppfd_total_expected = par_total * 4.6
        
        assert torch.isclose(ppfd_total_calc, ppfd_total_expected, rtol=1e-6)


class TestLAIProfileGeneration:
    """Test LAI vertical profile generation methods."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.radiation = RadiationFunctor()
    
    def test_uniform_lai_profile(self):
        """Test uniform LAI distribution."""
        lai_total = torch.tensor(4.0)
        n_layers = 20
        
        layer_lai = self.radiation._lai_profile(lai_total, n_layers)
        
        # Check shape and conservation
        assert layer_lai.shape == (n_layers,)
        assert torch.isclose(torch.sum(layer_lai), lai_total, rtol=1e-5)
        
        # Check uniformity
        expected_layer_lai = lai_total / n_layers
        assert torch.allclose(layer_lai, expected_layer_lai, rtol=1e-6)
    
    def test_beta_lai_profile(self, monkeypatch):
        """Test Beta distribution LAI profile."""
        radiation = _radiation_with_overrides(
            monkeypatch,
            lai_profile='beta',
            beta_params=[2.0, 2.0],
        )
        
        lai_total = torch.tensor(5.0)
        n_layers = 10
        
        layer_lai = radiation._lai_profile(lai_total, n_layers)
        
        # Check shape and conservation
        assert layer_lai.shape == (n_layers,)
        assert torch.isclose(torch.sum(layer_lai), lai_total, rtol=1e-4)
        
        # Check non-uniformity (Beta distribution should vary across layers)
        # For symmetric Beta(2,2), middle layers should have higher LAI
        max_lai = torch.max(layer_lai)
        min_lai = torch.min(layer_lai)
        assert max_lai > min_lai, "Beta profile should show variation across layers"
    
    def test_lai_profile_conservation(self, monkeypatch):
        """Test LAI conservation across different profiles."""
        lai_values = [1.0, 3.0, 6.0, 10.0]
        n_layers = 15
        
        for lai_val in lai_values:
            lai_total = torch.tensor(lai_val)
            
            # Test uniform profile
            layer_lai_uniform = self.radiation._lai_profile(lai_total, n_layers)
            assert torch.isclose(torch.sum(layer_lai_uniform), lai_total, rtol=1e-5)
            
            # Test beta profile
            radiation_beta = _radiation_with_overrides(
                monkeypatch,
                lai_profile='beta',
            )
            layer_lai_beta = radiation_beta._lai_profile(lai_total, n_layers)
            assert torch.isclose(torch.sum(layer_lai_beta), lai_total, rtol=1e-4)
    
    def test_lai_profile_non_negative(self):
        """Test LAI profile values are non-negative."""
        lai_total = torch.tensor(2.5)
        n_layers = 8
        
        layer_lai = self.radiation._lai_profile(lai_total, n_layers)
        
        assert torch.all(layer_lai >= 0.0), "LAI profile contains negative values"

    @pytest.mark.parametrize(
        ('beta_params', 'n_layers'),
        [
            ([2.0, 0.5], 10),
            ([0.5, 2.0], 10),
            ([0.5, 0.5], 10),
            ([2.0, 0.5], 1),
        ],
    )
    def test_beta_profiles_are_finite_for_endpoint_singular_shapes(
        self,
        monkeypatch,
        beta_params,
        n_layers,
    ):
        """Valid Beta shapes below one remain finite under midpoint quadrature."""
        radiation = _radiation_with_overrides(
            monkeypatch,
            lai_profile='beta',
            beta_params=beta_params,
        )
        lai_total = torch.tensor(4.0)

        layer_lai = radiation._lai_profile(lai_total, n_layers)

        assert layer_lai.shape == (n_layers,)
        assert torch.isfinite(layer_lai).all()
        assert torch.all(layer_lai >= 0.0)
        assert torch.sum(layer_lai).item() == pytest.approx(4.0, abs=1.0e-6)


class TestBeerLambertMultiLayer:
    """Test Beer-Lambert multi-layer radiation transfer calculations."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.radiation = RadiationFunctor()
    
    def test_beer_lambert_basic_calculation(self):
        """Test basic Beer-Lambert layer calculation."""
        par_dir = torch.tensor(200.0)  # W m^-2
        par_dif = torch.tensor(100.0)  # W m^-2  
        lai_total = torch.tensor(3.0)
        n_layers = 10
        
        result = self.radiation._beer_lambert_layers(par_dir, par_dif, lai_total, n_layers)
        
        # Check output structure
        assert 'apar_layer' in result
        assert 'fpar' in result
        assert 'transmission_dir' in result
        assert 'transmission_dif' in result
        
        # Check shapes
        assert result['apar_layer'].shape == (n_layers,)
        assert result['fpar'].numel() == 1
        
        # Check physical bounds
        assert 0.0 <= result['fpar'].item() <= 1.0
        assert torch.all(result['apar_layer'] >= 0.0)
    
    def test_energy_conservation(self):
        """Test energy conservation in Beer-Lambert model."""
        par_dir = torch.tensor(300.0)
        par_dif = torch.tensor(150.0)
        par_total = par_dir + par_dif
        lai_total = torch.tensor(4.0)
        n_layers = 20
        
        result = self.radiation._beer_lambert_layers(par_dir, par_dif, lai_total, n_layers)
        
        # Check conservation
        apar_total = torch.sum(result['apar_layer'])
        
        # Total absorbed should not exceed total available
        assert apar_total <= par_total + 1e-6, \
            f"Energy conservation violated: {apar_total} > {par_total}"
        
        # Check fPAR consistency
        expected_fpar = apar_total / par_total
        assert torch.isclose(result['fpar'], expected_fpar, rtol=1e-5)
    
    def test_lai_response(self):
        """Test radiation absorption response to LAI variation."""
        par_dir = torch.tensor(250.0)
        par_dif = torch.tensor(125.0)
        n_layers = 15
        
        lai_values = [0.5, 2.0, 4.0, 8.0]
        fpar_values = []
        
        for lai in lai_values:
            lai_tensor = torch.tensor(lai)
            result = self.radiation._beer_lambert_layers(par_dir, par_dif, lai_tensor, n_layers)
            fpar_values.append(result['fpar'].item())
        
        # fPAR should increase monotonically with LAI
        for i in range(1, len(fpar_values)):
            assert fpar_values[i] > fpar_values[i-1], \
                f"fPAR did not increase with LAI: {fpar_values}"
    
    def test_extinction_coefficient_effects(self, monkeypatch):
        """Test effects of different extinction coefficients."""
        par_dir = torch.tensor(200.0)
        par_dif = torch.tensor(100.0)
        lai_total = torch.tensor(3.0)
        n_layers = 10
        
        # High extinction coefficient
        radiation_high_k = _radiation_with_overrides(
            monkeypatch,
            k_b=0.8,
            k_d=1.0,
        )
        result_high_k = radiation_high_k._beer_lambert_layers(par_dir, par_dif, lai_total, n_layers)
        
        # Low extinction coefficient  
        radiation_low_k = _radiation_with_overrides(
            monkeypatch,
            k_b=0.3,
            k_d=0.5,
        )
        result_low_k = radiation_low_k._beer_lambert_layers(par_dir, par_dif, lai_total, n_layers)
        
        # Higher extinction coefficient should lead to higher absorption in upper layers
        apar_high_k = result_high_k['apar_layer']
        apar_low_k = result_low_k['apar_layer']
        
        # Check that high-k has more absorption in first few layers
        upper_layer_ratio = torch.sum(apar_high_k[:3]) / torch.sum(apar_low_k[:3])
        assert upper_layer_ratio > 1.0, "Higher extinction coefficient should increase upper layer absorption"

    def test_extinction_parameters_remain_differentiable(self, monkeypatch):
        """Trainable Beer-Lambert coefficients must retain autograd lineage."""
        radiation = _radiation_with_overrides(
            monkeypatch,
            k_b=0.6,
            k_d=0.8,
        )
        manager = radiation.get_parameter_manager()
        key_b = 'atmosphere.radiation.canopy.k_b'
        key_d = 'atmosphere.radiation.canopy.k_d'
        parameter_b = manager.nn_parameters[key_b]
        parameter_d = manager.nn_parameters[key_d]
        parameter_b.grad = None
        parameter_d.grad = None

        result = radiation._beer_lambert_layers(
            torch.tensor(200.0),
            torch.tensor(100.0),
            torch.tensor(3.0),
            10,
        )
        absorbed = result['apar_layer'].sum()

        assert absorbed.requires_grad
        absorbed.backward()
        for parameter in (parameter_b, parameter_d):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().item() > 0.0
    
    def test_zero_lai_case(self):
        """Test handling of zero LAI case."""
        par_dir = torch.tensor(180.0)
        par_dif = torch.tensor(90.0)
        lai_total = torch.tensor(0.0)
        n_layers = 5
        
        result = self.radiation._beer_lambert_layers(par_dir, par_dif, lai_total, n_layers)
        
        # No canopy means no absorption
        assert result['fpar'].item() == 0.0
        assert torch.all(result['apar_layer'] == 0.0)
        
        # All radiation should be transmitted
        assert torch.isclose(result['transmission_dir'], par_dir, rtol=1e-6)
        assert torch.isclose(result['transmission_dif'], par_dif, rtol=1e-6)


class TestConservationLaws:
    """Test conservation law enforcement and violation detection."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.radiation = RadiationFunctor()
    
    def test_par_conservation(self):
        """Test PAR conservation through the full calculation chain."""
        shortwave_down = torch.tensor(900.0)
        lai_total = torch.tensor(5.0)
        
        # Test full canopy calculation
        result = self.radiation._compute_canopy_radiation(
            shortwave_down=shortwave_down,
            extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
            lai_total=lai_total,
            return_diagnostics=True,
        )
        
        # Check PAR conservation
        par_total = result['diagnostics.radiation.par_energy']
        par_dir = result['diagnostics.radiation.par_direct']
        par_dif = result['diagnostics.radiation.par_diffuse']
        
        assert torch.isclose(par_total, par_dir + par_dif, rtol=1e-6)
        
        # Check absorbed + transmitted = incident
        apar_total = torch.sum(result['apar_layer'])
        fpar = result['fpar']
        
        expected_absorbed = fpar * par_total
        assert torch.isclose(apar_total, expected_absorbed, rtol=1e-5)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very high LAI
        par_dir = torch.tensor(100.0)
        par_dif = torch.tensor(50.0)
        lai_extreme = torch.tensor(20.0)  # Very high LAI
        n_layers = 30
        
        result = self.radiation._beer_lambert_layers(par_dir, par_dif, lai_extreme, n_layers)
        
        # Check all values are finite
        assert torch.isfinite(result['fpar']).all()
        assert torch.isfinite(result['apar_layer']).all()
        
        # Check physical bounds maintained
        assert 0.0 <= result['fpar'].item() <= 1.0
        assert torch.all(result['apar_layer'] >= 0.0)
    
    def test_fpar_monotonicity(self):
        """Test fPAR increases monotonically with LAI."""
        par_dir = torch.tensor(220.0)
        par_dif = torch.tensor(110.0)
        n_layers = 8
        
        lai_sequence = torch.linspace(0.1, 6.0, 20)
        fpar_sequence = []
        
        for lai in lai_sequence:
            result = self.radiation._beer_lambert_layers(par_dir, par_dif, lai, n_layers)
            fpar_sequence.append(result['fpar'].item())
        
        # Check monotonicity
        for i in range(1, len(fpar_sequence)):
            assert fpar_sequence[i] >= fpar_sequence[i-1], \
                f"fPAR not monotonic at LAI {lai_sequence[i]}: {fpar_sequence[i]} < {fpar_sequence[i-1]}"


class TestEdgeCasesAndBoundaryConditions:
    """Test model behavior under extreme and boundary conditions."""
    
    def setup_method(self):
        """Set up test configuration."""
        self.radiation = RadiationFunctor()
    
    def test_very_low_radiation(self):
        """Test behavior with very low incoming radiation."""
        shortwave_down = torch.tensor(5.0)  # Very low radiation
        lai_total = torch.tensor(2.0)
        
        result = self.radiation._compute_canopy_radiation(
            shortwave_down=shortwave_down,
            extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
            lai_total=lai_total,
        )
        
        # Should handle gracefully
        assert torch.isfinite(result['fpar']).all()
        assert torch.isfinite(result['apar_layer']).all()
        assert result['fpar'].item() >= 0.0
    
    def test_very_high_radiation(self):
        """Test behavior with very high incoming radiation."""
        shortwave_down = torch.tensor(1500.0)  # Very high radiation
        lai_total = torch.tensor(3.0)
        
        result = self.radiation._compute_canopy_radiation(
            shortwave_down=shortwave_down,
            extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
            lai_total=lai_total,
        )
        
        # Should handle gracefully without overflow
        assert torch.isfinite(result['fpar']).all()
        assert torch.isfinite(result['apar_layer']).all()
        assert _expected_par_energy(self.radiation, shortwave_down).item() > 0.0
    
    def test_single_layer_case(self, monkeypatch):
        """Test single layer canopy calculation."""
        radiation_single = _radiation_with_overrides(monkeypatch, layers=1)
        
        shortwave_down = torch.tensor(400.0)
        lai_total = torch.tensor(1.0)
        
        result = radiation_single._compute_canopy_radiation(
            shortwave_down=shortwave_down,
            extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
            lai_total=lai_total,
        )
        
        # Check single layer output
        assert result['apar_layer'].shape == (1,)
        assert torch.isfinite(result['apar_layer']).all()
        assert result['fpar'].item() > 0.0
    
    def test_many_layers_case(self, monkeypatch):
        """Test many layer canopy calculation.""" 
        radiation_many = _radiation_with_overrides(monkeypatch, layers=50)
        
        shortwave_down = torch.tensor(600.0)
        lai_total = torch.tensor(4.0)
        
        result = radiation_many._compute_canopy_radiation(
            shortwave_down=shortwave_down,
            extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
            lai_total=lai_total,
        )
        
        # Check many layers output
        assert result['apar_layer'].shape == (50,)
        assert torch.isfinite(result['apar_layer']).all()
        
        # Conservation should still hold
        apar_total = torch.sum(result['apar_layer'])
        par_total = _expected_par_energy(radiation_many, shortwave_down)
        assert apar_total <= par_total + 1e-5


class TestIntegrationAndconsistency:
    """Test integration with main radiation module and backward consistency."""
    
    def setup_method(self):
        """Set up test configurations."""
        self.standard_state = {
            'air_temperature': torch.tensor(298.15),
            'incoming_shortwave_potential': torch.tensor(1200.0),
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(295.15),
            'relative_humidity': torch.tensor(0.7),
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
            'leaf_area_index': torch.tensor(3.5)
        }
    
    def test_canopy_enabled_forward_integration(self):
        """Test canopy calculation integration in forward method."""
        radiation = RadiationFunctor()
        
        result = radiation.forward(self.standard_state, time=43200.0, dt=60.0)

        declared = set(radiation.metadata.output_variables)
        assert set(result) == declared
        
        # Should include both standard and canopy outputs
        standard_keys = ['shortwave_down', 'longwave_down', 'net_radiation', 'ppfd']
        canopy_keys = ['fpar', 'apar_layer']
        
        for key in standard_keys:
            assert key in result, f"Standard output '{key}' missing"
        
        for key in canopy_keys:
            assert key in result, f"Canopy output '{key}' missing when enabled"
            assert isinstance(result[key], UpdateItem)
            assert result[key].mode is UpdateMode.REPLACE
            assert result[key].per_timestep is False
        assert 'par' not in result
    
    def test_backward_consistency_when_disabled(self):
        """Test backward consistency when canopy is disabled."""
        was_enabled = feature_manager.is_enabled('canopy_layers_enabled')
        feature_manager.disable('canopy_layers_enabled')
        # Default configuration (canopy disabled)
        try:
            radiation_disabled = RadiationFunctor()
            result_disabled = radiation_disabled.forward(self.standard_state, time=43200.0, dt=60.0)

            # PPFD remains canonical while layer outputs stay disabled.
            assert 'ppfd' in result_disabled
            assert 'par' not in result_disabled
            canopy_keys = ['fpar', 'apar_layer']
            for key in canopy_keys:
                assert key not in result_disabled, f"Canopy output '{key}' found when disabled"
        finally:
            if was_enabled:
                feature_manager.enable('canopy_layers_enabled')
        
        # Should contain standard outputs
        standard_keys = ['shortwave_down', 'longwave_down', 'net_radiation', 'atmospheric_emissivity']
        for key in standard_keys:
            assert key in result_disabled, f"Standard output '{key}' missing when disabled"
    
    def test_diagnostics_output(self):
        """Test that declared diagnostic outputs are unconditional."""
        radiation = RadiationFunctor()
        
        result = radiation.forward(self.standard_state, time=43200.0, dt=60.0)

        assert set(result) == set(radiation.metadata.output_variables)
        
        # Should include diagnostic outputs
        diagnostic_keys = [
            'diagnostics.radiation.par_energy',
            'diagnostics.radiation.par_direct',
            'diagnostics.radiation.par_diffuse',
            'diagnostics.radiation.ppfd_direct',
            'diagnostics.radiation.ppfd_diffuse',
            'diagnostics.radiation.diffuse_fraction',
            'diagnostics.radiation.transmission_direct',
            'diagnostics.radiation.transmission_diffuse',
        ]
        for key in diagnostic_keys:
            assert key in result, f"Diagnostic output '{key}' missing"
            assert isinstance(result[key], UpdateItem)
    
    def test_lai_from_state_variable(self):
        """Test LAI retrieval from state variables."""
        radiation = RadiationFunctor()
        
        # Test with different LAI state variable names
        state_with_lai = self.standard_state.copy()
        state_with_lai['leaf_area_index'] = torch.tensor(2.5)
        
        result = radiation.forward(state_with_lai, time=43200.0, dt=60.0)
        
        # Should use LAI from state
        assert torch.isfinite(_value(result['fpar'])).all()
        assert _value(result['fpar']).item() > 0.0
    
    def test_selected_model_failure_is_not_repaired(self, monkeypatch):
        """A selected canopy model failure must terminate the forward call."""
        with pytest.raises(ValueError, match='outside bounds'):
            _radiation_with_overrides(monkeypatch, k_b=-1.0)

    def test_missing_lai_is_not_replaced_by_a_default(self):
        """Enabled canopy execution requires the declared LAI state cochain."""
        radiation = RadiationFunctor()
        state = self.standard_state.copy()
        del state['leaf_area_index']

        with pytest.raises(KeyError, match='leaf_area_index'):
            radiation.forward(state, time=43200.0, dt=60.0)


# Parametrized tests for different configurations
@pytest.mark.parametrize("n_layers", [5, 10, 20, 30])
def test_layer_count_scaling(n_layers, monkeypatch):
    """Test scaling with different layer counts."""
    radiation = _radiation_with_overrides(monkeypatch, layers=n_layers)
    
    shortwave_down = torch.tensor(500.0)
    lai_total = torch.tensor(3.0)
    
    result = radiation._compute_canopy_radiation(
        shortwave_down=shortwave_down,
        extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
        lai_total=lai_total,
    )
    
    # Check correct number of layers
    assert result['apar_layer'].shape == (n_layers,)
    
    # Check conservation regardless of layer count
    apar_total = torch.sum(result['apar_layer'])
    par_total = _expected_par_energy(radiation, shortwave_down)
    assert apar_total <= par_total + 1e-5


@pytest.mark.parametrize("lai_total", [0.5, 1.0, 2.0, 4.0, 6.0, 10.0])
def test_lai_range_response(lai_total):
    """Test response across LAI range.""" 
    radiation = RadiationFunctor()
    
    shortwave_down = torch.tensor(600.0)
    lai_tensor = torch.tensor(lai_total)
    
    result = radiation._compute_canopy_radiation(
        shortwave_down=shortwave_down,
        extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
        lai_total=lai_tensor,
    )
    
    # Check physical bounds
    assert 0.0 <= result['fpar'].item() <= 1.0
    assert torch.all(result['apar_layer'] >= 0.0)
    
    # Higher LAI should generally give higher fPAR (except for very high LAI saturation)
    if lai_total > 0.1:
        assert result['fpar'].item() > 0.0


def test_model_type_selection(monkeypatch):
    """Beer-Lambert is the sole honest canopy transfer implementation."""
    radiation = RadiationFunctor()
    shortwave_down = torch.tensor(400.0)
    lai_total = torch.tensor(2.5)
    result = radiation._compute_canopy_radiation(
        shortwave_down=shortwave_down,
        extraterrestrial_horizontal=_extraterrestrial_horizontal(shortwave_down),
        lai_total=lai_total,
    )

    assert set(result) == {'fpar', 'apar_layer'}

    assert 0.0 <= result['fpar'].item() <= 1.0
    assert torch.all(result['apar_layer'] >= 0.0)

    with pytest.raises(ValueError, match="Supported: 'beer_lambert'"):
        _radiation_with_overrides(monkeypatch, model_type='two_stream')
    assert not hasattr(RadiationFunctor, '_two_stream_layers')


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])


