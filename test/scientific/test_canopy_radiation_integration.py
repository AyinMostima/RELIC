"""
Test canopy multi-layer radiation integration with feature flags and energy conservation.

Tests the complete integration of canopy radiation functionality including:
- Global feature flag control
- Energy conservation validation 
- Multi-layer radiation transfer
- Interface with radiation energy conservation validator

Test Coverage:
- Feature flag integration (canopy_layers_enabled)
- Canonical Beer-Lambert radiation model and rejected proxy modes
- Energy conservation checks
- PAR/PPFD calculations
- Layer-wise absorption validation
"""

import pytest
import torch

# Import test targets
from relic.scientific.atmosphere.radiation import RadiationFunctor
from relic.scientific.atmosphere.radiation_energy_conservation import (
    RadiationEnergyConservationValidator,
)
from relic.core.feature_flags import feature_manager
from relic.core.state import UpdateItem


def _value(output):
    """Return the tensor payload from one explicit radiation update."""
    return output.value if isinstance(output, UpdateItem) else output


def _observed_scalar(output):
    """Detach a scalar test observation without altering the production graph."""
    value = _value(output)
    return value.detach().item() if torch.is_tensor(value) else float(value)


class TestCanopyRadiationIntegration:
    """Test suite for canopy radiation integration with feature flags and energy conservation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self._canopy_feature_was_enabled = feature_manager.is_enabled(
            'canopy_layers_enabled'
        )
        self.device = torch.device("cpu")
        feature_manager.enable('canopy_layers_enabled')
        self.radiation_functor = RadiationFunctor(
            name="test_radiation_canopy",
            device=self.device
        )
        
        # Sample state for testing
        self.test_state = {
            'air_temperature': torch.tensor(298.15),  # 25 degC
            'incoming_shortwave_potential': torch.tensor(800.0),  # W m^-2
            'cos_zenith': torch.tensor(0.8),
            'surface_temperature': torch.tensor(300.0),  # Slightly warmer surface
            'relative_humidity': torch.tensor(0.6),
            'surface_albedo': torch.tensor(0.15),
            'surface_emissivity': torch.tensor(0.95),
            'cloud_fraction': torch.tensor(0.3),
            'leaf_area_index': torch.tensor(3.5)  # Moderate LAI for canopy
        }
        self.energy_validator = None

    def teardown_method(self):
        """Restore the process-wide feature flag after every test."""
        if self._canopy_feature_was_enabled:
            feature_manager.enable('canopy_layers_enabled')
        else:
            feature_manager.disable('canopy_layers_enabled')
    def test_global_feature_flag_control(self):
        """Test that global feature flag controls canopy radiation."""
        feature_manager.disable('canopy_layers_enabled')
        disabled_functor = RadiationFunctor(name='test_radiation_disabled')
        fluxes = disabled_functor.forward(
            state=self.test_state,
            time=43200.0,  # Noon
            dt=3600.0
        )
        
        # Should have basic radiation outputs but no canopy-specific outputs
        assert 'shortwave_down' in fluxes
        assert 'net_radiation' in fluxes
        
        canopy_outputs = ['fpar', 'apar_layer']
        for output in canopy_outputs:
            assert output not in fluxes
        assert 'ppfd' in fluxes
        assert 'par' not in fluxes

        feature_manager.enable('canopy_layers_enabled')
        enabled_functor = RadiationFunctor(name='test_radiation_enabled')
        fluxes_enabled = enabled_functor.forward(
            state=self.test_state,
            time=43200.0,
            dt=3600.0,
        )
        
        # Should now have canopy outputs
        for output in canopy_outputs:
            assert output in fluxes_enabled, f"Missing canopy output: {output}"
        
        # Verify PAR energy diagnostics and canonical PPFD are reasonable.
        par_value = _observed_scalar(fluxes_enabled['diagnostics.radiation.par_energy'])
        ppfd_value = _observed_scalar(fluxes_enabled['ppfd'])
        fpar_value = _observed_scalar(fluxes_enabled['fpar'])
        
        assert 0 < par_value < 500, f"PAR value out of range: {par_value}"
        assert 0 < ppfd_value < 2500, f"PPFD value out of range: {ppfd_value}"
        assert 0 < fpar_value < 1, f"fPAR value out of range: {fpar_value}"
        
        print("Global feature flag control: PASS")
        
    def test_canopy_energy_conservation(self):
        """Test energy conservation in canopy radiation calculations."""
        # Enable canopy
        feature_manager.enable('canopy_layers_enabled')
        
        # Run radiation with diagnostics
        fluxes = self.radiation_functor.forward(
            state=self.test_state,
            time=43200.0,
            dt=3600.0,
        )
        
        # Extract canopy radiation components
        par_total = _value(fluxes['diagnostics.radiation.par_energy'])
        apar_layer = _value(fluxes['apar_layer'])
        fpar = _value(fluxes['fpar'])
        transmitted = (
            _value(fluxes['diagnostics.radiation.transmission_direct'])
            + _value(fluxes['diagnostics.radiation.transmission_diffuse'])
        )
        
        # Basic conservation: sum of layer absorption <= total PAR
        apar_sum = torch.sum(apar_layer)
        par_total_value = _observed_scalar(par_total)
        apar_sum_value = _observed_scalar(apar_sum)
        transmitted_value = _observed_scalar(transmitted)
        fpar_value = _observed_scalar(fpar)
        conservation_error = abs(
            apar_sum_value + transmitted_value - par_total_value
        ) / par_total_value
        
        print(f"PAR total: {par_total_value:.3f} W m^-2")
        print(f"APAR sum: {apar_sum_value:.3f} W m^-2")
        print(f"Conservation error: {conservation_error:.6f}")
        print(f"fPAR calculated: {fpar_value:.3f}")
        
        assert conservation_error < 1.0e-6, f"Energy conservation violated: {conservation_error}"
        
        # fPAR consistency check
        fpar_from_layers = apar_sum_value / par_total_value if par_total_value > 0 else 0
        fpar_error = abs(fpar_value - fpar_from_layers)
        assert fpar_error < 1e-6, f"fPAR inconsistency: {fpar_error}"
        
        # Physical bounds checking
        assert torch.all(apar_layer >= 0), "Negative layer absorption detected"
        assert 0 <= fpar_value <= 1, f"fPAR out of physical bounds: {fpar_value}"
        
        print("Canopy energy conservation: PASS")
        
    def test_energy_conservation_validator_integration(self):
        """Test integration with energy conservation validator."""
        # Enable canopy
        feature_manager.enable('canopy_layers_enabled')
        
        # Run radiation calculation
        fluxes = self.radiation_functor.forward(
            state=self.test_state,
            time=43200.0,
            dt=3600.0,
        )
        
        if self.energy_validator is None:
            self.energy_validator = RadiationEnergyConservationValidator()
        raw_fluxes = {
            key: _value(fluxes[key]).detach()
            for key in (
                'shortwave_down',
                'shortwave_net',
                'longwave_down',
                'longwave_net',
                'net_radiation',
            )
        }
        raw_fluxes['longwave_up'] = _value(
            fluxes['diagnostics.radiation.longwave_up']
        ).detach()
        canopy_outputs = {
            'par_total': _value(
                fluxes['diagnostics.radiation.par_energy']
            ).detach(),
            'apar_layer': _value(fluxes['apar_layer']).detach(),
            'fpar': _value(fluxes['fpar']).detach(),
        }
        conservation_report = self.energy_validator.validate_energy_conservation(
            radiation_fluxes=raw_fluxes,
            radiation_state={
                'surface_albedo': self.test_state['surface_albedo'],
                'surface_emissivity': self.test_state['surface_emissivity'],
                'surface_temperature': self.test_state['surface_temperature'],
            },
            canopy_outputs=canopy_outputs,
        )

        assert torch.isfinite(
            torch.tensor(conservation_report.energy_balance_error)
        )
        assert conservation_report.canopy_balance['apar_sum'] <= _observed_scalar(
            canopy_outputs['par_total']
        )
        assert conservation_report.canopy_balance['fpar'] == pytest.approx(
            _observed_scalar(canopy_outputs['fpar'])
        )
        
    def test_beer_lambert_is_unique_and_proxy_model_is_rejected(self):
        """The production graph exposes only the evidence-backed Beer-Lambert model."""
        feature_manager.enable('canopy_layers_enabled')
        beer_lambert_functor = RadiationFunctor(
            name="test_beer_lambert",
        )
        
        beer_fluxes = beer_lambert_functor.forward(
            state=self.test_state,
            time=43200.0,
            dt=3600.0
        )
        
        beer_fpar = _observed_scalar(beer_fluxes['fpar'])
        assert 0 < beer_fpar < 1, f"Beer-Lambert fPAR out of range: {beer_fpar}"
        with pytest.raises(
            ValueError,
            match='Unsupported RadiationFunctor config keys: canopy',
        ):
            RadiationFunctor(
                name="test_two_stream",
                config={'canopy': {'model_type': {'value': 'two_stream'}}},
            )
        assert not hasattr(RadiationFunctor, '_two_stream_layers')
        
    def test_layer_wise_energy_conservation(self):
        """Test detailed layer-wise energy conservation."""
        feature_manager.enable('canopy_layers_enabled')
        
        detailed_functor = RadiationFunctor(
            name="test_detailed_layers",
        )
        
        fluxes = detailed_functor.forward(
            state=self.test_state,
            time=43200.0,
            dt=3600.0,
        )
        
        par_total = _observed_scalar(fluxes['diagnostics.radiation.par_energy'])
        apar_layer = _value(fluxes['apar_layer'])
        transmitted = _observed_scalar(
            _value(fluxes['diagnostics.radiation.transmission_direct'])
            + _value(fluxes['diagnostics.radiation.transmission_diffuse'])
        )
        n_layers = len(apar_layer)
        
        print(f"Testing {n_layers} canopy layers")
        print(f"Total PAR: {par_total:.3f} W m^-2")
        
        # Check each layer
        layer_sum = 0.0
        for i, layer_apar in enumerate(apar_layer):
            layer_value = _observed_scalar(layer_apar)
            layer_sum += layer_value
            
            # Each layer should have non-negative absorption
            assert layer_value >= 0, f"Layer {i} has negative absorption: {layer_value}"
            
            # Each layer should not exceed total PAR
            assert layer_value <= par_total, f"Layer {i} exceeds total PAR: {layer_value}"
        
        # Total should match
        conservation_error = abs(layer_sum + transmitted - par_total) / par_total
        print(f"Layer sum: {layer_sum:.3f} W m^-2")
        print(f"Conservation error: {conservation_error:.2e}")
        
        assert conservation_error < 1.0e-6, f"Layer-wise conservation error: {conservation_error}"
        
        print("Layer-wise energy conservation: PASS")
        
    def test_feature_flag_is_single_topology_owner(self):
        """Only the global feature flag may select the canopy topology."""
        with pytest.raises(
            ValueError,
            match='Unsupported RadiationFunctor config keys: canopy',
        ):
            RadiationFunctor(
                name='test_local_enabled',
                config={'canopy': {'enabled': {'value': True}}},
            )

        feature_manager.disable('canopy_layers_enabled')
        disabled_functor = RadiationFunctor(name='test_canopy_disabled')
        disabled_fluxes = disabled_functor.forward(
            state=self.test_state,
            time=43200.0,
            dt=3600.0,
        )
        assert 'fpar' not in disabled_fluxes
        assert 'ppfd' in disabled_fluxes

        feature_manager.enable('canopy_layers_enabled')
        enabled_functor = RadiationFunctor(name='test_canopy_enabled')
        enabled_fluxes = enabled_functor.forward(
            state=self.test_state,
            time=43200.0,
            dt=3600.0,
        )
        assert 'fpar' in enabled_fluxes
        assert 'ppfd' in enabled_fluxes
        assert 'par' not in enabled_fluxes


def test_factory_function():
    """Test radiation energy conservation validator factory function."""
    validator = RadiationEnergyConservationValidator()
    assert validator is not None

