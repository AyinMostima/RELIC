"""
Comprehensive L1 Unit Tests for Atmospheric Radiation Model

This test suite provides exhaustive testing of the RadiationFunctor class,
covering all radiation transfer calculations, energy balance components, edge cases,
and scientific accuracy. Tests ensure zero-defect operation and validate against
established literature values for radiation physics.

Test Categories:
1. Basic Functionality Tests
2. Shortwave Radiation Calculation Tests  
3. Longwave Radiation Calculation Tests
4. Energy Balance and Conservation Tests
5. Scientific Accuracy Validation
6. Edge Case and Boundary Tests
7. Error Handling and Robustness Tests

References for validation values:
- Allen et al. (1998). FAO Irrigation Paper 56
- Brutsaert (1975). On a derivable formula for long-wave radiation  
- Campbell & Norman (1998). Environmental Biophysics
- Monteith & Unsworth (2013). Principles of Environmental Physics
"""

from dataclasses import replace

import pytest
import torch

from relic.core.feature_flags import feature_manager
from relic.core.simulator import RELICSimulator
from relic.core.state import StateWriteMode, UpdateItem, UpdateMode
from relic.parameters import UnifiedParameterManager, get_global_manager
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.utils.physical_constants import get_physical_constant


CANONICAL_OUTPUT_UNITS = {
    'shortwave_down': 'W m-2',
    'longwave_down': 'W m-2',
    'net_radiation': 'W m-2',
    'shortwave_net': 'W m-2',
    'longwave_net': 'W m-2',
    'atmospheric_emissivity': 'dimensionless',
    'ppfd': 'umol photons m-2 s-1',
}

RADIATION_DIAGNOSTIC_UNITS = {
    'diagnostics.radiation.par_energy': 'W m-2',
    'diagnostics.radiation.par_direct': 'W m-2',
    'diagnostics.radiation.par_diffuse': 'W m-2',
    'diagnostics.radiation.ppfd_direct': 'umol photons m-2 s-1',
    'diagnostics.radiation.ppfd_diffuse': 'umol photons m-2 s-1',
    'diagnostics.radiation.diffuse_fraction': 'dimensionless',
    'diagnostics.radiation.longwave_up': 'W m-2',
    'diagnostics.radiation.longwave_emitted': 'W m-2',
    'diagnostics.radiation.longwave_reflected': 'W m-2',
    'diagnostics.radiation.surface_albedo_used': 'dimensionless',
    'diagnostics.radiation.surface_emissivity_used': 'dimensionless',
    'diagnostics.radiation.cos_zenith_used': 'dimensionless',
    'diagnostics.radiation.vapor_pressure_calculated': 'Pa',
    'diagnostics.radiation.shortwave_clipping_residual': 'W m-2',
    'diagnostics.radiation.emissivity_clipping_residual': 'dimensionless',
}


def _value(output):
    """Return the tensor payload while migration tests assert wrappers explicitly."""
    return output.value if isinstance(output, UpdateItem) else output


def _manager_with_trainable_value(key, value):
    """Inject one calibrated value through the canonical ParameterManager path."""
    manager = UnifiedParameterManager()
    spec = manager.get_parameter(key)
    assert spec is not None
    manager.register_parameter(replace(spec, value=value))
    return manager


class TestRadiationBasicFunctionality:
    """Test basic functionality and interface compliance for RadiationFunctor."""
    
    @pytest.fixture
    def device(self):
        """Test device (CPU or CUDA if available)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_initialization_default_parameters(self, device):
        """Test radiation functor initialization with default parameters."""
        radiation = RadiationFunctor(device=device)
        
        # Check basic attributes
        assert radiation.name == "atmospheric_radiation"
        assert hasattr(radiation, 'metadata')
        assert hasattr(radiation, 'config')
        
        # Verify parameters are loaded
        assert radiation.get_parameter_value('atmospheric_transmittance') is not None
        assert radiation.get_parameter_value('brutsaert_coefficient') is not None
        assert radiation.get_parameter_value('cloud_attenuation_factor') is not None
        
        # Test parameter bounds - handle dict format
        transmittance_raw = radiation.get_parameter_value('atmospheric_transmittance')
        transmittance = transmittance_raw.get('value', transmittance_raw) if isinstance(transmittance_raw, dict) else transmittance_raw
        assert 0.4 <= transmittance <= 1.0
        
        brutsaert_coef_raw = radiation.get_parameter_value('brutsaert_coefficient')
        brutsaert_coef = brutsaert_coef_raw.get('value', brutsaert_coef_raw) if isinstance(brutsaert_coef_raw, dict) else brutsaert_coef_raw
        assert 1.0 <= brutsaert_coef <= 1.5
        
    def test_scientific_parameters_use_injected_manager_only(self, device):
        """Radiation accepts manager injection and rejects local parameter copies."""
        manager = get_global_manager()
        radiation = RadiationFunctor(parameter_manager=manager, device=device)

        assert radiation.get_parameter_manager() is manager
        with pytest.raises(
            ValueError,
            match='Unsupported RadiationFunctor config keys: parameters',
        ):
            RadiationFunctor(
                config={
                    'parameters': {
                        'atmospheric_transmittance': {'value': 0.8},
                    }
                }
            )

    def test_parameter_units_are_validated_against_the_runtime_contract(
        self,
        monkeypatch,
    ):
        """A YAML unit drift must fail before a radiation coefficient is used."""
        manager = get_global_manager()
        original_get_parameter = manager.get_parameter
        target_key = 'atmosphere.radiation.atmospheric_transmittance'

        def _get_parameter_with_unit_drift(key, namespace=None):
            spec = original_get_parameter(key, namespace=namespace)
            if key == target_key and spec is not None:
                return replace(spec, units='W m-2')
            return spec

        monkeypatch.setattr(manager, 'get_parameter', _get_parameter_with_unit_drift)

        with pytest.raises(
            ValueError,
            match=(
                'Cannot convert bounds for '
                'atmosphere.radiation.atmospheric_transmittance from W m-2 '
                'to required units dimensionless'
            ),
        ):
            RadiationFunctor(parameter_manager=manager)

    def test_disabled_canopy_does_not_read_layer_only_parameters(self, monkeypatch):
        """The disabled canopy topology must not resolve unused layer parameters."""
        manager = get_global_manager()
        original_get_parameter = manager.get_parameter
        accessed_keys = set()
        layer_only_keys = {
            'atmosphere.radiation.canopy.model_type',
            'atmosphere.radiation.canopy.layers',
            'atmosphere.radiation.canopy.k_b',
            'atmosphere.radiation.canopy.k_d',
            'atmosphere.radiation.canopy.lai_profile',
            'atmosphere.radiation.canopy.beta_params',
        }

        def _record_parameter_access(key, namespace=None):
            accessed_keys.add(key)
            return original_get_parameter(key, namespace=namespace)

        monkeypatch.setattr(manager, 'get_parameter', _record_parameter_access)
        was_enabled = feature_manager.is_enabled('canopy_layers_enabled')
        feature_manager.disable('canopy_layers_enabled')
        try:
            RadiationFunctor(parameter_manager=manager)
        finally:
            if was_enabled:
                feature_manager.enable('canopy_layers_enabled')

        assert layer_only_keys.isdisjoint(accessed_keys)

    def test_radiation_does_not_read_environmental_vpd_parameters(self, monkeypatch):
        """Radiation may derive actual vapor pressure but must not own the VPD law."""
        manager = get_global_manager()
        original_get_parameter = manager.get_parameter
        accessed_keys = set()

        def _record_parameter_access(key, namespace=None):
            accessed_keys.add(key)
            return original_get_parameter(key, namespace=namespace)

        monkeypatch.setattr(manager, 'get_parameter', _record_parameter_access)
        radiation = RadiationFunctor(parameter_manager=manager)
        state = {
            'air_temperature': torch.tensor(298.15),
            'incoming_shortwave_potential': torch.tensor(1000.0),
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(295.15),
            'relative_humidity': torch.tensor(0.7),
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
        }

        outputs = radiation.forward(state, time=43200.0, dt=60.0)

        environmental_vpd_prefix = 'atmosphere.environmental_forcing.vpd.'
        assert not any(
            key.startswith(environmental_vpd_prefix) for key in accessed_keys
        )
        assert not any(
            key.startswith(environmental_vpd_prefix)
            for key in radiation.required_parameters()
        )
        assert 'vapor_pressure_deficit' not in outputs
        assert 'vpd' not in outputs

    @pytest.mark.parametrize(
        ('config', 'exception_type', 'message'),
        [
            (['canopy'], TypeError, 'config must be a dictionary'),
            ({'unknown_mode': True}, ValueError, 'Unsupported RadiationFunctor config keys'),
            (
                {'parameters': {'unknown_parameter': {'value': 1.0}}},
                ValueError,
                'Unsupported RadiationFunctor config keys: parameters',
            ),
            (
                {'canopy': {'unknown_parameter': {'value': 1.0}}},
                ValueError,
                'Unsupported RadiationFunctor config keys: canopy',
            ),
            (
                {'canopy': {'model_type': {'value': 'two_stream'}}},
                ValueError,
                'Unsupported RadiationFunctor config keys: canopy',
            ),
        ],
    )
    def test_invalid_configuration_fails_fast(self, config, exception_type, message):
        """Invalid configuration must not be silently replaced or ignored."""
        with pytest.raises(exception_type, match=message):
            RadiationFunctor(config=config)

    def test_metadata_tracks_selected_canopy_outputs(self):
        """Metadata declares canopy outputs only for the selected canopy path."""
        was_enabled = feature_manager.is_enabled('canopy_layers_enabled')
        try:
            feature_manager.disable('canopy_layers_enabled')
            base = RadiationFunctor()
            feature_manager.enable('canopy_layers_enabled')
            canopy = RadiationFunctor()
        finally:
            if was_enabled:
                feature_manager.enable('canopy_layers_enabled')
            else:
                feature_manager.disable('canopy_layers_enabled')

        assert set(CANONICAL_OUTPUT_UNITS).issubset(base.metadata.output_variables)
        assert set(RADIATION_DIAGNOSTIC_UNITS).issubset(base.metadata.output_variables)
        assert {'fpar', 'apar_layer'}.isdisjoint(base.metadata.output_variables)
        assert {'fpar', 'apar_layer'}.issubset(canopy.metadata.output_variables)
        assert 'par' not in base.metadata.output_variables
        assert 'par' not in canopy.metadata.output_variables
        assert 'leaf_area_index' not in base.metadata.input_variables
        assert 'leaf_area_index' in canopy.metadata.input_variables
        assert 'enabled' not in base._canopy_params
        assert 'canopy.enabled' not in base.metadata.parameters

    def test_removed_two_stream_branch_has_no_runtime_callable(self):
        """The former Beer-Lambert proxy must not remain as a selectable branch."""
        assert not hasattr(RadiationFunctor, '_two_stream_layers')
        
    def test_device_consistency(self, device):
        """Test CPU/GPU device consistency."""
        radiation = RadiationFunctor(device=device)
        
        # Check device is set correctly
        if device.type == 'cuda':
            assert radiation.device.type == 'cuda'
        else:
            assert radiation.device.type == 'cpu'
            
    def test_dependencies_interface(self, device):
        """Test get_dependencies interface compliance."""
        radiation = RadiationFunctor(device=device)
        dependencies = radiation.get_dependencies()
        
        assert isinstance(dependencies, list)
        # Radiation typically has no dependencies
        assert len(dependencies) == 0
        
    def test_functor_metadata(self, device):
        """Test functor metadata is properly set."""
        radiation = RadiationFunctor(device=device)
        metadata = radiation.metadata
        
        assert metadata is not None
        assert metadata.name == "atmospheric_radiation"
        assert "radiation transfer" in metadata.description.lower()
        assert "net_radiation" in metadata.output_variables
        assert metadata.input_variables == [
            'air_temperature',
            'incoming_shortwave_potential',
            'cos_zenith',
            'surface_temperature',
            'relative_humidity',
            'surface_albedo',
            'surface_emissivity',
            'cloud_fraction',
        ]


class TestShortwaveRadiationCalculations:
    """Test shortwave radiation calculation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.radiation = RadiationFunctor()
        
    def test_clear_sky_shortwave_calculation(self):
        """Test clear-sky shortwave radiation calculation."""
        # Standard test conditions
        cos_zenith = torch.tensor(0.8)  # High sun angle
        sw_potential = torch.tensor(1200.0)  # W m^-2, typical extraterrestrial
        cloud_fraction = None  # Clear sky
        
        sw_down = self.radiation._calculate_shortwave_radiation(
            cos_zenith, sw_potential, cloud_fraction
        )
        
        # Check reasonable magnitude
        assert sw_down.item() > 0
        assert sw_down.item() < sw_potential.item()  # Must be less than extraterrestrial
        
        # Check atmospheric transmission effects - handle dict format
        transmittance_raw = self.radiation.get_parameter_value('atmospheric_transmittance')
        transmittance = transmittance_raw.get('value', transmittance_raw) if isinstance(transmittance_raw, dict) else transmittance_raw
        expected_sw = sw_potential * transmittance * cos_zenith
        assert torch.isclose(sw_down, expected_sw, rtol=0.05)
        
    def test_cloudy_sky_shortwave_calculation(self):
        """Test cloudy-sky shortwave radiation calculation."""
        cos_zenith = torch.tensor(0.8)
        sw_potential = torch.tensor(1200.0)
        cloud_fraction = torch.tensor(0.5)  # 50% cloud cover
        
        sw_down = self.radiation._calculate_shortwave_radiation(
            cos_zenith, sw_potential, cloud_fraction
        )
        
        # Clear-sky reference
        sw_clear = self.radiation._calculate_shortwave_radiation(
            cos_zenith, sw_potential, None
        )
        
        # Cloudy should be less than clear sky
        assert sw_down < sw_clear
        
    def test_zenith_angle_dependency(self):
        """Test solar zenith angle dependency."""
        sw_potential = torch.tensor(1200.0)
        zenith_angles = [0.2, 0.5, 0.8, 1.0]  # cos(zenith) values
        sw_values = []
        
        for cos_zenith in zenith_angles:
            sw_down = self.radiation._calculate_shortwave_radiation(
                torch.tensor(cos_zenith), sw_potential, None
            )
            sw_values.append(sw_down.item())
            
        # Should increase with cos(zenith) (higher sun angle)
        for i in range(1, len(sw_values)):
            assert sw_values[i] > sw_values[i-1]
            
    def test_nighttime_conditions(self):
        """Test zero radiation during nighttime (cos_zenith <= 0)."""
        sw_potential = torch.tensor(1200.0)
        assert (
            'atmosphere.radiation.minimum_cos_zenith'
            not in self.radiation.required_parameters()
        )

        for cosine in (-1.0, -0.2, 0.0):
            sw_down = self.radiation._calculate_shortwave_radiation(
                torch.tensor(cosine),
                sw_potential,
                None,
            )
            assert sw_down.item() == 0.0


class TestLongwaveRadiationCalculations:
    """Test longwave radiation calculation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.radiation = RadiationFunctor()
        
    def test_brutsaert_atmospheric_emissivity(self):
        """Test Brutsaert atmospheric emissivity formula."""
        air_temp = torch.tensor(298.15)  # 25 degC
        surface_temp = torch.tensor(295.15)  # 22 degC
        vapor_pressure = torch.tensor(2000.0)  # Pa, moderate humidity
        surface_emissivity = torch.tensor(0.97)
        
        lw_components = self.radiation._calculate_longwave_radiation(
            air_temp, surface_temp, vapor_pressure, surface_emissivity
        )
        
        # Check atmospheric emissivity is reasonable
        epsilon_atm = lw_components['atmospheric_emissivity']
        assert 0.4 <= epsilon_atm.item() <= 1.0
        
        # Check longwave downward is positive and reasonable
        lw_down = lw_components['longwave_down']
        assert 200.0 <= lw_down.item() <= 500.0  # Typical range
        
    def test_longwave_energy_balance(self):
        """Test longwave radiation energy balance components."""
        air_temp = torch.tensor(298.15)
        surface_temp = torch.tensor(295.15) 
        vapor_pressure = torch.tensor(2000.0)
        surface_emissivity = torch.tensor(0.97)
        
        lw_components = self.radiation._calculate_longwave_radiation(
            air_temp, surface_temp, vapor_pressure, surface_emissivity
        )
        
        lw_down = lw_components['longwave_down']
        lw_up = lw_components['longwave_up']
        lw_net = lw_components['longwave_net']
        
        # Check energy balance consistency
        expected_lw_net = lw_down - lw_up
        assert torch.isclose(lw_net, expected_lw_net, rtol=1e-6)
        
        # All components should be positive
        assert lw_down.item() > 0
        assert lw_up.item() > 0
        
    def test_temperature_sensitivity(self):
        """Test longwave radiation temperature sensitivity."""
        surface_temp = torch.tensor(295.15)
        vapor_pressure = torch.tensor(2000.0)
        surface_emissivity = torch.tensor(0.97)
        
        # Test air temperature sensitivity
        temps = [288.15, 298.15, 308.15]  # 15 degC, 25 degC, 35 degC
        lw_down_values = []
        
        for temp in temps:
            lw_components = self.radiation._calculate_longwave_radiation(
                torch.tensor(temp), surface_temp, vapor_pressure, surface_emissivity
            )
            lw_down_values.append(lw_components['longwave_down'].item())
            
        # Longwave down should increase with air temperature (T^4 relationship)
        for i in range(1, len(lw_down_values)):
            assert lw_down_values[i] > lw_down_values[i-1]
            
    def test_cloud_enhancement_longwave(self):
        """Test cloud enhancement of longwave radiation."""
        air_temp = torch.tensor(298.15)
        surface_temp = torch.tensor(295.15)
        vapor_pressure = torch.tensor(2000.0)
        surface_emissivity = torch.tensor(0.97)
        
        # Clear sky
        lw_clear = self.radiation._calculate_longwave_radiation(
            air_temp, surface_temp, vapor_pressure, surface_emissivity, None
        )
        
        # Cloudy sky
        cloud_fraction = torch.tensor(0.8)
        lw_cloudy = self.radiation._calculate_longwave_radiation(
            air_temp, surface_temp, vapor_pressure, surface_emissivity, cloud_fraction
        )
        
        # Clouds should enhance longwave down
        assert lw_cloudy['longwave_down'] > lw_clear['longwave_down']


class TestEnergyBalanceAndConservation:
    """Test energy balance components and conservation laws."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.radiation = RadiationFunctor()
        self.standard_state = {
            'air_temperature': torch.tensor(298.15),
            'incoming_shortwave_potential': torch.tensor(1200.0),
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(295.15),
            'relative_humidity': torch.tensor(0.7),
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
        }

    def test_forward_has_exact_canonical_outputs_and_replace_semantics(self):
        """Default execution returns the complete declared instantaneous contract."""
        result = self.radiation.forward(
            self.standard_state,
            time=43200.0,
            dt=60.0,
        )

        expected_units = CANONICAL_OUTPUT_UNITS | RADIATION_DIAGNOSTIC_UNITS
        assert set(result) == set(self.radiation.metadata.output_variables)
        assert set(result) == set(expected_units)
        for key, unit in expected_units.items():
            update = result[key]
            assert isinstance(update, UpdateItem), key
            assert update.mode is UpdateMode.REPLACE, key
            assert update.per_timestep is False, key
            assert update.unit == unit, key
            assert update.source_functor == 'atmospheric_radiation', key
            assert torch.isfinite(update.value).all(), key

    def test_ppfd_is_unconditional_without_lai_and_matches_energy_conversion(self):
        """PPFD is derived from shortwave_down even with canopy disabled and no LAI."""
        was_enabled = feature_manager.is_enabled('canopy_layers_enabled')
        feature_manager.disable('canopy_layers_enabled')
        radiation = RadiationFunctor()
        state = dict(self.standard_state)
        transmittance = radiation.get_parameter_value(
            'atmospheric_transmittance',
            namespace='atmosphere.radiation',
            source_order=('manager',),
            required=True,
        )
        if isinstance(transmittance, torch.Tensor):
            transmittance = transmittance.detach().item()
        state['incoming_shortwave_potential'] = torch.tensor(500.0 / transmittance)
        state['cos_zenith'] = torch.tensor(1.0)
        state['cloud_fraction'] = torch.tensor(0.0)
        try:
            result = radiation.forward(state, time=43200.0, dt=60.0)
        finally:
            if was_enabled:
                feature_manager.enable('canopy_layers_enabled')

        shortwave_down = _value(result['shortwave_down'])
        ppfd = _value(result['ppfd'])
        assert 'leaf_area_index' not in state
        assert 'par' not in result
        assert shortwave_down.item() == pytest.approx(500.0, abs=1.0e-6)
        assert ppfd.item() == pytest.approx(1035.0, abs=1.0e-4)
        expected_ppfd = (
            shortwave_down
            * radiation._get_canopy_parameter('par_fraction_of_sw')
            * radiation._get_canopy_parameter('ppfd_conversion')
        )
        assert ppfd.item() == pytest.approx(
            expected_ppfd.detach().item(),
            abs=1.0e-4,
        )

    def test_diagnostics_are_namespaced_tensors_with_explicit_replace_semantics(self):
        """Diagnostics use one unconditional namespaced, typed output contract."""
        result = self.radiation.forward(
            self.standard_state,
            time=43200.0,
            dt=60.0,
        )

        assert set(result) == set(CANONICAL_OUTPUT_UNITS) | set(
            RADIATION_DIAGNOSTIC_UNITS
        )
        forbidden = {
            'par',
            'par_dir',
            'par_dif',
            'ppfd_dir',
            'ppfd_dif',
            'diffuse_fraction',
            'longwave_up',
            'surface_albedo_used',
            'surface_emissivity_used',
            'cos_zenith_used',
            'vapor_pressure_calculated',
            'energy_conservation_report',
            'shortwave_clipping_residual',
            'emissivity_clipping_residual',
        }
        assert forbidden.isdisjoint(result)
        for key, unit in RADIATION_DIAGNOSTIC_UNITS.items():
            update = result[key]
            assert isinstance(update, UpdateItem), key
            assert update.mode is UpdateMode.REPLACE, key
            assert update.per_timestep is False, key
            assert update.unit == unit, key
            assert update.source_functor == 'atmospheric_radiation', key
            assert torch.isfinite(update.value).all(), key
        
    def test_net_radiation_calculation(self):
        """Test net radiation calculation and components."""
        result = self.radiation.forward(self.standard_state, time=43200.0, dt=60.0)
        
        # Check all expected outputs are present
        required_outputs = ['shortwave_down', 'longwave_down', 'net_radiation', 
                          'shortwave_net', 'longwave_net']
        for output in required_outputs:
            assert output in result
            assert torch.is_tensor(_value(result[output]))
            
        # Check energy balance consistency
        sw_net = _value(result['shortwave_net'])
        lw_net = _value(result['longwave_net'])
        net_radiation = _value(result['net_radiation'])
        
        expected_rn = sw_net + lw_net
        assert torch.isclose(net_radiation, expected_rn, rtol=1e-6)
        
    def test_shortwave_balance_components(self):
        """Test shortwave radiation balance components."""
        result = self.radiation.forward(self.standard_state, time=43200.0, dt=60.0)
        
        sw_down = _value(result['shortwave_down'])
        sw_net = _value(result['shortwave_net'])
        albedo = self.standard_state['surface_albedo']
        
        # SW_net = (1 - albedo) * SW_down
        expected_sw_net = (1.0 - albedo) * sw_down
        assert torch.isclose(sw_net, expected_sw_net, rtol=1e-6)
        
    def test_physical_bounds_validation(self):
        """Test physical bounds of radiation components."""
        result = self.radiation.forward(self.standard_state, time=43200.0, dt=60.0)
        
        # Shortwave down should be positive during daytime
        assert _value(result['shortwave_down']).item() >= 0
        
        # Longwave components should be positive
        assert _value(result['longwave_down']).item() > 0
        
        # Net radiation magnitude should be reasonable
        rn = _value(result['net_radiation']).item()
        assert -200.0 <= rn <= 800.0  # Typical range for daytime


class TestScientificAccuracyValidation:
    """Test against established literature values and benchmarks."""
    
    def setup_method(self):
        """Set up standard test conditions."""
        self.radiation = RadiationFunctor()
        
        # Standard noon conditions for validation (Allen et al. 1998)
        self.reference_conditions = {
            'air_temperature': torch.tensor(298.15),        # 25 degC
            'incoming_shortwave_potential': torch.tensor(1200.0),  # W/m2
            'cos_zenith': torch.tensor(1.0),                # Noon, overhead sun
            'surface_temperature': torch.tensor(298.15),   # 25 degC
            'relative_humidity': torch.tensor(0.70),       # 70%
            'surface_albedo': torch.tensor(0.20),          # Typical vegetation
            'surface_emissivity': torch.tensor(0.97),      # Typical vegetation
            'cloud_fraction': torch.tensor(0.0)            # Clear sky
        }
        
    def test_clear_sky_radiation_magnitudes(self):
        """Test radiation magnitudes against literature values."""
        result = self.radiation.forward(self.reference_conditions, time=43200.0, dt=60.0)
        
        sw_down = _value(result['shortwave_down']).item()
        lw_down = _value(result['longwave_down']).item()
        net_radiation = _value(result['net_radiation']).item()
        
        # Literature ranges for clear-sky conditions at solar noon
        assert 600.0 <= sw_down <= 1000.0, f"SW down {sw_down} outside expected range [600-1000]"
        assert 300.0 <= lw_down <= 450.0, f"LW down {lw_down} outside expected range [300-450]"  
        assert 400.0 <= net_radiation <= 700.0, f"Net radiation {net_radiation} outside expected range [400-700]"
        
    def test_brutsaert_emissivity_accuracy(self):
        """Test Brutsaert atmospheric emissivity against literature."""
        air_temp = torch.tensor(298.15)
        vapor_pressure = torch.tensor(2000.0)  # Pa
        
        # Calculate using internal method
        surface_temp = torch.tensor(295.15)
        surface_emissivity = torch.tensor(0.97)
        
        lw_result = self.radiation._calculate_longwave_radiation(
            air_temp, surface_temp, vapor_pressure, surface_emissivity
        )
        
        epsilon_atm = lw_result['atmospheric_emissivity'].item()
        
        # Literature range for these conditions: 0.7-0.8 (Brutsaert 1975)
        assert 0.65 <= epsilon_atm <= 0.85, f"Atmospheric emissivity {epsilon_atm} outside expected range"
        
    def test_diurnal_cycle_response(self):
        """Test radiation diurnal cycle characteristics."""
        cos_zenith_values = [0.0, 0.3, 0.6, 0.9, 1.0]  # Dawn to noon
        sw_down_values = []
        
        for cos_zenith in cos_zenith_values:
            conditions = self.reference_conditions.copy()
            conditions['cos_zenith'] = torch.tensor(cos_zenith)
            
            result = self.radiation.forward(conditions, time=43200.0, dt=60.0)
            sw_down_values.append(_value(result['shortwave_down']).item())
            
        # Should increase monotonically from dawn to noon
        for i in range(1, len(sw_down_values)):
            assert sw_down_values[i] >= sw_down_values[i-1]
            
        # First value (cos_zenith=0) should be very low or zero
        assert sw_down_values[0] <= 10.0
        
    def test_albedo_response_accuracy(self):
        """Test surface albedo effects on radiation balance."""
        albedo_values = [0.1, 0.2, 0.3, 0.5]  # Different surface types
        sw_net_values = []
        
        for albedo in albedo_values:
            conditions = self.reference_conditions.copy()
            conditions['surface_albedo'] = torch.tensor(albedo)
            
            result = self.radiation.forward(conditions, time=43200.0, dt=60.0)
            sw_net_values.append(_value(result['shortwave_net']).item())
            
        # SW_net should decrease with increasing albedo
        for i in range(1, len(sw_net_values)):
            assert sw_net_values[i] < sw_net_values[i-1]


class TestEdgeCasesAndBoundaryConditions:
    """Test model behavior under extreme and boundary conditions."""
    
    def setup_method(self):
        """Set up test model."""
        self.radiation = RadiationFunctor()
        
    def test_nighttime_conditions(self):
        """Test radiation during nighttime (negative cos_zenith)."""
        night_state = {
            'air_temperature': torch.tensor(285.15),       # Cool night
            'incoming_shortwave_potential': torch.tensor(0.0),  # No extraterrestrial
            'cos_zenith': torch.tensor(-0.5),              # Below horizon
            'surface_temperature': torch.tensor(283.15),
            'relative_humidity': torch.tensor(0.8),
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
        }
        
        result = self.radiation.forward(night_state, time=0.0, dt=60.0)
        
        # Shortwave should be zero or very small at night
        assert _value(result['shortwave_down']).item() <= 1.0
        assert _value(result['shortwave_net']).item() <= 1.0
        
        # Longwave should still be positive
        assert _value(result['longwave_down']).item() > 0
        
        # Net radiation should be negative (radiative cooling)
        assert _value(result['net_radiation']).item() < 0
        
    def test_very_high_albedo(self):
        """Test behavior with very high surface albedo."""
        high_albedo_state = {
            'air_temperature': torch.tensor(298.15),
            'incoming_shortwave_potential': torch.tensor(1200.0),
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(295.15),
            'relative_humidity': torch.tensor(0.7),
            'surface_albedo': torch.tensor(0.9),           # Very high albedo (snow/ice)
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
        }
        
        result = self.radiation.forward(high_albedo_state, time=43200.0, dt=60.0)
        
        # Net shortwave should be very low
        sw_net = _value(result['shortwave_net']).item()
        sw_down = _value(result['shortwave_down']).item()
        expected_sw_net = 0.1 * sw_down  # (1 - 0.9) * SW_down
        
        assert torch.isclose(torch.tensor(sw_net), torch.tensor(expected_sw_net), rtol=0.01)
        
    def test_extreme_temperature_conditions(self):
        """Test behavior at extreme temperatures."""
        # Very hot conditions
        hot_state = {
            'air_temperature': torch.tensor(323.15),       # 50 degC
            'incoming_shortwave_potential': torch.tensor(1400.0),
            'cos_zenith': torch.tensor(0.9),
            'surface_temperature': torch.tensor(328.15),   # 55 degC surface
            'relative_humidity': torch.tensor(0.3),
            'surface_albedo': torch.tensor(0.15),          # Dark surface
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
        }
        
        self.radiation.forward(hot_state, time=43200.0, dt=60.0)
        
        # Longwave up should be very high due to T^4 relationship
        lw_components = self.radiation._calculate_longwave_radiation(
            hot_state['air_temperature'], hot_state['surface_temperature'],
            torch.tensor(1000.0), hot_state['surface_emissivity']
        )
        
        assert lw_components['longwave_up'].item() > 600.0  # Very high emission
        
    def test_full_cloud_cover(self):
        """Test behavior under full cloud cover."""
        cloudy_state = {
            'air_temperature': torch.tensor(298.15),
            'incoming_shortwave_potential': torch.tensor(1200.0),
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(295.15),
            'relative_humidity': torch.tensor(0.9),        # High humidity
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(1.0)            # Full cloud cover
        }
        
        result = self.radiation.forward(cloudy_state, time=43200.0, dt=60.0)
        
        # Compare with clear sky
        clear_state = cloudy_state.copy()
        clear_state['cloud_fraction'] = torch.tensor(0.0)
        clear_result = self.radiation.forward(clear_state, time=43200.0, dt=60.0)
        
        # Cloudy SW should be less than clear
        assert _value(result['shortwave_down']) < _value(clear_result['shortwave_down'])
        
        # Cloudy LW should be greater than clear
        assert _value(result['longwave_down']) > _value(clear_result['longwave_down'])
        

class TestErrorHandlingAndRobustness:
    """Test error handling and robustness under various conditions."""
    
    def setup_method(self):
        """Set up test model."""
        self.radiation = RadiationFunctor()
        self.valid_state = {
            'air_temperature': torch.tensor(298.15),
            'incoming_shortwave_potential': torch.tensor(1200.0),
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(295.15),
            'relative_humidity': torch.tensor(0.7),
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
        }
        
    @pytest.mark.parametrize(
        'missing_key',
        [
            'air_temperature',
            'incoming_shortwave_potential',
            'cos_zenith',
            'surface_temperature',
            'relative_humidity',
            'surface_albedo',
            'surface_emissivity',
            'cloud_fraction',
        ],
    )
    def test_missing_required_inputs(self, missing_key):
        """Every declared state cochain is required by the production path."""
        incomplete_state = self.valid_state.copy()
        del incomplete_state[missing_key]

        with pytest.raises(KeyError, match=missing_key):
            self.radiation.forward(incomplete_state, time=43200.0, dt=60.0)

    def test_out_of_range_temperatures(self):
        """Test handling of extreme temperature values."""
        extreme_state = self.valid_state.copy()
        extreme_state['air_temperature'] = torch.tensor(400.15)

        with pytest.raises(ValueError, match='air_temperature'):
            self.radiation.forward(extreme_state, time=43200.0, dt=60.0)
            
    def test_invalid_albedo_values(self):
        """Test handling of invalid albedo values."""
        invalid_state = self.valid_state.copy()
        invalid_state['surface_albedo'] = torch.tensor(1.5)
        
        with pytest.raises(ValueError, match="albedo"):
            self.radiation.forward(invalid_state, time=43200.0, dt=60.0)
            
    def test_nan_input_handling(self):
        """Test handling of NaN inputs."""
        nan_state = self.valid_state.copy()
        nan_state['air_temperature'] = torch.tensor(float('nan'))

        with pytest.raises(ValueError, match='non-finite'):
            self.radiation.forward(nan_state, time=43200.0, dt=60.0)
            
    def test_tensor_vs_scalar_inputs(self):
        """Test handling of mixed tensor/scalar inputs."""
        mixed_state = self.valid_state.copy()
        mixed_state['air_temperature'] = 298.15

        with pytest.raises(TypeError, match='air_temperature'):
            self.radiation.forward(mixed_state, time=43200.0, dt=60.0)

    def test_nonpositive_dt_is_rejected(self):
        """The production lifecycle must not silently accept a disabled step."""
        with pytest.raises(ValueError, match='dt must be positive'):
            self.radiation.forward(self.valid_state, time=43200.0, dt=0.0)

    def test_scalar_tensor_dt_matches_float_runtime(self):
        """Simulator scalar-tensor time steps use the same radiation closure."""
        float_result = self.radiation.forward(
            self.valid_state,
            time=43200.0,
            dt=60.0,
        )
        tensor_result = self.radiation.forward(
            self.valid_state,
            time=43200.0,
            dt=torch.tensor(60.0),
        )

        assert set(tensor_result) == set(float_result)
        for key in float_result:
            torch.testing.assert_close(
                tensor_result[key].value,
                float_result[key].value,
                rtol=0.0,
                atol=0.0,
            )

    def test_unexpected_runtime_mode_is_rejected(self):
        """Forward rejects hidden execution and conditional-output modes."""
        with pytest.raises(TypeError, match='Unsupported RadiationFunctor.forward options'):
            self.radiation.forward(
                self.valid_state,
                time=43200.0,
                dt=60.0,
                legacy_mode=True,
            )
        with pytest.raises(TypeError, match='Unsupported RadiationFunctor.forward options'):
            self.radiation.forward(
                self.valid_state,
                time=43200.0,
                dt=60.0,
                return_diagnostics=True,
            )


class TestParameterSensitivity:
    """Test parameter sensitivity and configuration effects."""
    
    def setup_method(self):
        """Set up test conditions."""
        self.base_conditions = {
            'air_temperature': torch.tensor(298.15),
            'incoming_shortwave_potential': torch.tensor(1200.0),
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(295.15),
            'relative_humidity': torch.tensor(0.7),
            'surface_albedo': torch.tensor(0.2),
            'surface_emissivity': torch.tensor(0.97),
            'cloud_fraction': torch.tensor(0.0),
        }
        
    def test_transmittance_sensitivity(self):
        """Test sensitivity to atmospheric transmittance parameter."""
        # Default model
        model_default = RadiationFunctor()
        result_default = model_default.forward(self.base_conditions, time=43200.0, dt=60.0)
        
        # High transmittance model through the canonical manager.
        manager = _manager_with_trainable_value(
            'atmosphere.radiation.atmospheric_transmittance',
            0.84,
        )
        model_high = RadiationFunctor(parameter_manager=manager)
        result_high = model_high.forward(self.base_conditions, time=43200.0, dt=60.0)
        
        # Higher transmittance should give higher shortwave down
        assert _value(result_high['shortwave_down']) > _value(result_default['shortwave_down'])
        
    def test_brutsaert_coefficient_sensitivity(self):
        """Test sensitivity to Brutsaert coefficient."""
        # Default model
        model_default = RadiationFunctor()
        result_default = model_default.forward(self.base_conditions, time=43200.0, dt=60.0)
        
        # High coefficient model through the canonical manager.
        manager = _manager_with_trainable_value(
            'atmosphere.radiation.brutsaert_coefficient',
            1.4,
        )
        model_high = RadiationFunctor(parameter_manager=manager)
        result_high = model_high.forward(self.base_conditions, time=43200.0, dt=60.0)
        
        # Higher coefficient should give higher longwave down
        assert _value(result_high['longwave_down']) > _value(result_default['longwave_down'])

    def test_transmittance_parameter_remains_differentiable(self):
        """A trainable optical parameter must remain on the autograd graph."""
        key = 'atmosphere.radiation.atmospheric_transmittance'
        manager = _manager_with_trainable_value(key, 0.8)
        parameter = manager.nn_parameters[key]
        parameter.grad = None
        model = RadiationFunctor(parameter_manager=manager)

        shortwave_down = _value(
            model.forward(
                self.base_conditions,
                time=43200.0,
                dt=60.0,
            )['shortwave_down']
        )

        assert shortwave_down.requires_grad
        shortwave_down.backward()
        assert parameter.grad is not None
        assert parameter.grad.item() == pytest.approx(960.0, rel=1.0e-6)


# Integration with pytest fixtures and parameterized tests
@pytest.fixture(scope="module")
def radiation_model():
    """Pytest fixture for Radiation model."""
    return RadiationFunctor()


@pytest.fixture(scope="module")
def standard_conditions():
    """Pytest fixture for standard test conditions."""
    return {
        'air_temperature': torch.tensor(298.15),
        'incoming_shortwave_potential': torch.tensor(1200.0),
        'cos_zenith': torch.tensor(0.8),
        'surface_temperature': torch.tensor(295.15),
        'relative_humidity': torch.tensor(0.7),
        'surface_albedo': torch.tensor(0.2),
        'surface_emissivity': torch.tensor(0.97),
        'cloud_fraction': torch.tensor(0.0),
    }


@pytest.mark.parametrize("cos_zenith", [0.2, 0.5, 0.8, 1.0])
def test_zenith_angle_sweep(radiation_model, standard_conditions, cos_zenith):
    """Parametrized test across solar zenith angle range."""
    conditions = standard_conditions.copy()
    conditions['cos_zenith'] = torch.tensor(cos_zenith)
    
    result = radiation_model.forward(conditions, time=43200.0, dt=60.0)
    
    # All outputs should be finite and physically reasonable
    assert torch.isfinite(_value(result['net_radiation']))
    assert _value(result['shortwave_down']).item() >= 0.0
    assert _value(result['longwave_down']).item() > 0.0


@pytest.mark.parametrize("albedo", [0.1, 0.2, 0.5, 0.8])
def test_albedo_sweep(radiation_model, standard_conditions, albedo):
    """Parametrized test across surface albedo range."""
    conditions = standard_conditions.copy()
    conditions['surface_albedo'] = torch.tensor(albedo)
    
    result = radiation_model.forward(conditions, time=43200.0, dt=60.0)
    
    assert torch.isfinite(_value(result['net_radiation']))
    assert torch.isfinite(_value(result['shortwave_net']))
    
    # Higher albedo should reduce net shortwave
    sw_net = _value(result['shortwave_net']).item()
    sw_down = _value(result['shortwave_down']).item()
    expected_sw_net = (1.0 - albedo) * sw_down
    assert abs(sw_net - expected_sw_net) < 1.0  # Small tolerance for numerical precision


@pytest.mark.parametrize("temperature", [283.15, 298.15, 313.15])
def test_temperature_sweep(radiation_model, standard_conditions, temperature):
    """Parametrized test across temperature range."""
    conditions = standard_conditions.copy()
    conditions['air_temperature'] = torch.tensor(temperature)
    conditions['surface_temperature'] = torch.tensor(temperature)
    
    result = radiation_model.forward(conditions, time=43200.0, dt=60.0)
    
    assert torch.isfinite(_value(result['net_radiation']))
    assert _value(result['longwave_down']).item() > 0.0
    
    # Longwave should increase with temperature
    if temperature > 290.0:  # Only test for reasonable temperatures
        assert _value(result['longwave_down']).item() > 200.0


@pytest.mark.parametrize("emissivity", [0.0, 0.97, 1.0])
def test_graybody_longwave_independent_physical_oracle(standard_conditions, emissivity):
    """A reciprocal opaque surface absorbs and emits on the same LW basis."""
    radiation = RadiationFunctor()
    state = {name: value.double() for name, value in standard_conditions.items()}
    state['surface_emissivity'] = torch.tensor(emissivity, dtype=torch.float64)
    state['cos_zenith'] = torch.tensor(-0.5, dtype=torch.float64)
    outputs = radiation(state, time=0.0, dt=60.0)
    incident = _value(outputs['longwave_down'])
    sigma = get_physical_constant('stefan_boltzmann_constant')
    # References: Noah-MP Technical Note, Eqs. 3.6.330 and 3.6.416-417;
    #   Nicodemus (1965), doi:10.1364/AO.4.000767.
    # Mathematical expression: E=eps*sigma*Ts^4; F=(1-eps)*Ldown;
    #   Lup=E+F; Lnet=eps*Ldown-E; Rnet=SWnet+Lnet.
    # Parameter meanings: eps surface LW emissivity/absorptivity; sigma SSOT
    #   Stefan-Boltzmann constant; Ts surface temperature; E emitted and
    #   F reflected LW; Ldown incident LW; Lup total outgoing LW.
    # Units: fluxes [W m-2]; sigma [W m-2 K-4]; Ts [K]; eps [1].
    # Domain/constraints: reciprocal opaque grey surface, finite scalar
    #   inputs, eps in [0,1], night SW=0; use independent component equations.
    # Sign convention: Ldown/Lnet/Rnet inward-positive; E/F/Lup outward-positive.
    emitted = emissivity * sigma * state['surface_temperature'] ** 4
    reflected = (1.0 - emissivity) * incident
    net = emissivity * incident - emitted
    torch.testing.assert_close(_value(outputs['longwave_net']), net, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(_value(outputs['diagnostics.radiation.longwave_up']), emitted + reflected)
    torch.testing.assert_close(_value(outputs['net_radiation']), net, rtol=1e-6, atol=1e-6)
    for component, expected in (('emitted', emitted), ('reflected', reflected)):
        update = outputs['diagnostics.radiation.longwave_' + component]
        assert update.write_mode is StateWriteMode.DERIVED
        assert update.mode is UpdateMode.REPLACE and update.per_timestep is False
        assert update.unit == 'W m-2'
        torch.testing.assert_close(update.value, expected)
    if emissivity == 0.0:
        assert net.item() == 0.0
    else:
        assert net.item() < 0.0


def test_graybody_autograd_retains_surface_and_manager_dependencies(standard_conditions):
    """Absorptivity scales the actual Brutsaert binding and surface derivatives."""
    key = 'atmosphere.radiation.brutsaert_coefficient'
    manager = _manager_with_trainable_value(key, 1.24)
    radiation = RadiationFunctor(parameter_manager=manager)
    state = {name: value.double() for name, value in standard_conditions.items()}
    emissivity = state['surface_emissivity'].detach().clone().requires_grad_()
    temperature = state['surface_temperature'].detach().clone().requires_grad_()
    state['surface_emissivity'], state['surface_temperature'] = emissivity, temperature
    outputs = radiation(state, time=43200.0, dt=60.0)
    incident = _value(outputs['longwave_down'])
    net = _value(outputs['longwave_net'])
    binding = manager.nn_parameters[key]
    grad_eps, grad_temp, grad_binding = torch.autograd.grad(
        net, (emissivity, temperature, binding), retain_graph=True,
    )
    grad_incident_binding = torch.autograd.grad(incident, binding)[0]
    sigma = get_physical_constant('stefan_boltzmann_constant')
    # References: Noah-MP Eq. 3.6.330; Brutsaert (1975); native PyTorch autograd.
    # Mathematical expression: dLnet/deps=Ldown-sigma*Ts^4;
    #   dLnet/dTs=-4*eps*sigma*Ts^3; dLnet/dC1=eps*dLdown/dC1.
    # Parameter meanings: eps surface emissivity; Ts surface temperature;
    #   C1 manager-owned atmospheric emissivity coefficient; sigma SSOT constant.
    # Units: eps/C1 [1]; Ts [K]; flux [W m-2]; temperature derivative [W m-2 K-1].
    # Domain/constraints: finite interior inputs and unclipped atmospheric
    #   emissivity; compare the actual governed tensor graph, not copied bindings.
    # Sign convention: positive derivative increases inward net LW.
    torch.testing.assert_close(grad_eps, incident.detach() - sigma * temperature.detach() ** 4)
    torch.testing.assert_close(grad_temp, -4.0 * emissivity.detach() * sigma * temperature.detach() ** 3)
    torch.testing.assert_close(grad_binding, emissivity.detach().to(binding) * grad_incident_binding)
    assert grad_binding.item() != 0.0


def test_graybody_outputs_commit_through_real_state_manager(standard_conditions):
    """The simulator commits corrected fluxes once and keeps components diagnostic."""
    simulator = RELICSimulator({'device': 'cpu', 'time_step': 60.0, 'auto_report': False})
    simulator.register_functor(RadiationFunctor(), execution_group='fast')
    initial = {name: value.double() for name, value in standard_conditions.items()}
    simulator.initialize_state(initial)
    outputs = simulator.step(dt=60.0)
    committed = simulator.state_manager.get_state()
    sigma = get_physical_constant('stefan_boltzmann_constant')
    # References: Noah-MP Eq. 3.6.330; existing StateManager REPLACE contract.
    # Mathematical expression: Lnet_committed=eps*Ldown-eps*sigma*Ts^4;
    #   Rnet_committed=SWnet+Lnet_committed; flux REPLACE never multiplies by dt.
    # Parameter meanings: eps/Ts original accepted forcing; sigma SSOT constant;
    #   dt simulator interval; Ldown/SWnet actual committed directional fluxes.
    # Units: fluxes [W m-2]; sigma [W m-2 K-4]; Ts [K]; eps [1]; dt [s].
    # Domain/constraints: governed simulator object call and one real StateManager
    #   commit with canonical writer; component diagnostics are not stored state.
    # Sign convention: net LW/Rn inward-positive, cooling remains negative.
    expected = initial['surface_emissivity'] * (
        committed['longwave_down'] - sigma * initial['surface_temperature'] ** 4
    )
    torch.testing.assert_close(committed['longwave_net'], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(committed['net_radiation'], committed['shortwave_net'] + expected,
                               rtol=1e-6, atol=1e-6)
    for key in ('diagnostics.radiation.longwave_emitted', 'diagnostics.radiation.longwave_reflected'):
        assert key not in committed
        assert key in outputs
    records = simulator.get_variable_provenance('longwave_net')
    assert records and all(record['functor'] == 'atmospheric_radiation' for record in records)
    assert simulator.current_time == 60.0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])


