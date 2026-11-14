"""
L3 System Tests for Photosynthesis Workflow

Tests complete photosynthesis workflow including parameter loading,
environmental coupling, computation, and conservation checks.
"""

import pytest
import torch
import numpy as np

from relic.scientific.biosphere.photosynthesis.farquhar import FarquharPhotosynthesis
from relic.parameters import get_global_manager
from relic.core.state import StateManager, UpdateItem


def _canonical_state(
    *,
    air_temperature=298.15,
    atmospheric_co2_ppm=400.0,
    ppfd=1000.0,
    vapor_pressure_deficit=1000.0,
    device=None,
    **optional,
):
    """Build required forcing ports and optional Farquhar controls."""
    state = {
        "air_temperature": torch.as_tensor(air_temperature, device=device),
        "atmospheric_co2_ppm": torch.as_tensor(atmospheric_co2_ppm, device=device),
        "ppfd": torch.as_tensor(ppfd, device=device),
        "leaf_area_index": torch.tensor(1.0),
        "vapor_pressure_deficit": torch.as_tensor(
            vapor_pressure_deficit, device=device
        ),
    }
    state.update(
        {name: torch.as_tensor(value, device=device) for name, value in optional.items()}
    )
    return state


def _value(output, key):
    """Return the tensor payload from an explicit state update."""
    assert isinstance(output[key], UpdateItem)
    return output[key].value


class TestPhotosynthesisSystemWorkflow:
    """Test complete photosynthesis system workflow."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()
        self.device = torch.device('cpu')
        
    def test_complete_workflow_execution(self):
        """Test complete workflow from environment to photosynthesis output."""
        # Initialize components
        state_manager = StateManager()
        photosynthesis = FarquharPhotosynthesis(device=self.device)
        
        # Set initial environmental conditions
        initial_conditions = _canonical_state(
            air_temperature=298.15,
            atmospheric_co2_ppm=410.0,
            ppfd=800.0,
            vapor_pressure_deficit=1000.0,
            water_stress_factor=1.0,
            device=self.device,
        )
        
        # Initialize state
        for key, value in initial_conditions.items():
            state_manager.set_state(key, value)

        state = state_manager.get_state()
        photo_output = photosynthesis.forward(state, time=0.0, dt=3600.0)

        state_manager.apply_updates(
            photo_output, dt=3600.0, check_conservation=False
        )
                
        # Verify workflow completion
        gpp = state_manager.get_state('gross_photosynthesis')
        assert gpp is not None, "GPP not in final state"
        
        # Verify reasonable values
        gpp_value = gpp.item()
        assert 0 <= gpp_value <= 100, f"GPP out of range: {gpp_value}"
        
        # Verify energy conservation (simplified)
        # Net photosynthesis should be less than gross
        net = state_manager.get_state('net_photosynthesis')
        # Reserved diagnostics.* outputs are derived records, never state.
        rd = state_manager.get_diagnostics()["current"][
            'diagnostics.photosynthesis.dark_respiration'
        ]
        
        net_value = net.item()
        rd_value = float(rd)
        
        assert net_value <= gpp_value, "Net photosynthesis exceeds gross"
        assert rd_value >= 0, "Dark respiration is negative"
        assert abs((gpp_value - rd_value) - net_value) < 0.001, \
            "Carbon balance violation: GPP - Rd != Net"
            
    def test_diurnal_cycle_simulation(self):
        """Test photosynthesis over a diurnal cycle."""
        state_manager = StateManager()
        photosynthesis = FarquharPhotosynthesis(device=self.device)
        
        # Simulate 24 hours with hourly timesteps
        hours = np.arange(0, 24)
        
        # Simple diurnal PPFD pattern (umol photons m-2 s-1).
        ppfd_max = 2000.0
        ppfd_values = ppfd_max * np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
        
        # Temperature pattern (K), kept constant to isolate the PPFD response.
        temp_values = np.full_like(hours, 298.15, dtype=float)
                     
        results = {
            'hour': hours,
            'ppfd': [],
            'temperature': [],
            'gpp': [],
            'net': [],
            'respiration': []
        }
        
        for hour, ppfd, temp in zip(hours, ppfd_values, temp_values):
            # Set state for this hour
            state = _canonical_state(
                air_temperature=temp,
                ppfd=ppfd,
                device=self.device,
            )
            
            # Run photosynthesis
            output = photosynthesis.forward(state, time=hour*3600, dt=3600.0)
            
            # Store results
            results['ppfd'].append(ppfd)
            results['temperature'].append(temp - 273.15)  # Convert to C
            results['gpp'].append(_value(output, 'gross_photosynthesis').item())
            results['net'].append(_value(output, 'net_photosynthesis').item())
            results['respiration'].append(
                _value(output, 'diagnostics.photosynthesis.dark_respiration').item()
            )
            
        # Verify diurnal patterns
        results['ppfd'] = np.array(results['ppfd'])
        results['gpp'] = np.array(results['gpp'])
        results['net'] = np.array(results['net'])
        
        # GPP should correlate with PPFD.
        ppfd_threshold = 0.1 * ppfd_max
        ppfd_nonzero = results['ppfd'] > ppfd_threshold
        if ppfd_nonzero.any():
            # Use manual correlation to avoid MKL OpenMP conflicts in numpy.corrcoef.
            ppfd_series = results['ppfd'][ppfd_nonzero]
            gpp_series = results['gpp'][ppfd_nonzero]
            ppfd_mean = float(np.mean(ppfd_series))
            gpp_mean = float(np.mean(gpp_series))
            ppfd_centered = ppfd_series - ppfd_mean
            gpp_centered = gpp_series - gpp_mean
            denom = float(np.sqrt(np.mean(ppfd_centered ** 2)) * np.sqrt(np.mean(gpp_centered ** 2)))
            correlation = float(np.mean(ppfd_centered * gpp_centered) / denom) if denom > 0 else 0.0
            assert correlation > 0.8, f"GPP-PPFD correlation too low: {correlation}"
            
        # GPP should be zero at night
        night_hours = (hours < 6) | (hours > 18)
        night_gpp = results['gpp'][night_hours]
        assert np.all(night_gpp < 1.0), "Significant GPP at night"
        
        # Peak GPP should occur during midday
        peak_hour = np.argmax(results['gpp'])
        assert 8 <= peak_hour <= 16, f"Peak GPP at unusual hour: {peak_hour}"
        
    def test_stress_response_cascade(self):
        """Test cascading stress effects through the system."""
        photosynthesis = FarquharPhotosynthesis(device=self.device)
        
        # Base state
        base_state = _canonical_state(
            ppfd=1500.0,
            device=self.device,
        )
        
        # Test different stress conditions
        stress_scenarios = [
            ('normal', {}),
            ('water_stress_factor', {'water_stress_factor': 0.5}),
            ('temperature_stress_factor', {'temperature_stress_factor': 0.6}),
            ('combined_stress', {
                'water_stress_factor': 0.5,
                'temperature_stress_factor': 0.6,
            }),
            ('ozone_stress_factor', {'ozone_stress_factor': 0.7}),
            ('all_stress', {
                'water_stress_factor': 0.5,
                'temperature_stress_factor': 0.6,
                'ozone_stress_factor': 0.7,
            })
        ]
        
        results = {}
        
        for scenario_name, stress_factors in stress_scenarios:
            state = base_state.copy()
            state.update({k: torch.tensor(v, device=self.device) 
                         for k, v in stress_factors.items()})
            
            output = photosynthesis.forward(state, time=0.0, dt=3600.0)
            results[scenario_name] = _value(
                output, 'gross_photosynthesis'
            ).item()
            
        # Verify stress effects
        assert results['water_stress_factor'] < results['normal'], \
            "Water stress didn't reduce GPP"
        assert results['temperature_stress_factor'] < results['normal'], \
            "Temperature stress didn't reduce GPP"
        assert results['ozone_stress_factor'] < results['normal'], \
            "Ozone stress didn't reduce GPP"
        assert results['combined_stress'] < results['water_stress_factor'], \
            "Combined stress not worse than single stress"
        assert results['all_stress'] < results['combined_stress'], \
            "All stresses not worse than partial stresses"
            
        # Verify multiplicative stress (approximately)
        expected_all = results['normal'] * 0.5 * 0.6 * 0.7  # All stress factors
        actual_all = results['all_stress']
        # Allow 10% tolerance for numerical differences
        assert abs(actual_all - expected_all) / expected_all < 0.1, \
            f"Stress multiplication incorrect: expected {expected_all:.2f}, got {actual_all:.2f}"
            
    def test_parameter_sensitivity_analysis(self):
        """Test system sensitivity to the manager-held reference capacities."""
        photosynthesis = FarquharPhotosynthesis(device=self.device)
        bindings = self.param_manager.get_trainable_parameters()

        def sensitivity(name, ppfd):
            binding = bindings[f"biosphere.photosynthesis.{name}"]
            assert getattr(photosynthesis, name) is binding
            # References: Farquhar et al. (1980); architecture SSOT section 4.2.
            # Mathematical expression: S = d A_net / d theta on the executed graph.
            # Parameter meanings: A_net is net canopy uptake; theta is the
            #   manager-owned reference capacity Vcmax25, Jmax25 or Rd25; S its
            #   sensitivity.
            # Units: A_net [umol CO2 m-2 ground s-1]; Vcmax25, Rd25 [umol CO2 m-2
            #   leaf s-1]; Jmax25 [umol e- m-2 leaf s-1]; S in A_net per theta units.
            # Domain/constraints: fixed valid C3 forcing, Rubisco-limited at PPFD 2000
            #   and electron-transport-limited at PPFD 1000 umol photons m-2 s-1; no
            #   parameter replacement, value update or extra process calculation.
            # Sign convention: positive S means greater theta raises net uptake.
            output = photosynthesis(
                _canonical_state(ppfd=ppfd, device=self.device), time=0.0, dt=1.0
            )
            gradient, = torch.autograd.grad(
                _value(output, "net_photosynthesis").sum(), (binding,)
            )
            return gradient

        sensitivities = {
            'vcmax25': sensitivity('vcmax25', 2000.0),
            'jmax25': sensitivity('jmax25', 1000.0),
            'rd25': sensitivity('rd25', 1000.0),
        }

        assert sensitivities['vcmax25'].item() > 0
        assert sensitivities['jmax25'].item() > 0
        assert sensitivities['rd25'].item() < 0
            
    def test_conservation_through_workflow(self):
        """Test that conservation principles are maintained through workflow."""
        state_manager = StateManager()
        photosynthesis = FarquharPhotosynthesis(device=self.device)
        
        # Initial state with known carbon content
        initial_state = _canonical_state(
            ppfd=600.0,
            device=self.device,
        )
        
        for key, value in initial_state.items():
            state_manager.set_state(key, value)
            
        # Track carbon fluxes
        carbon_fluxes = []
        
        # Run for multiple timesteps
        dt = 3600.0  # 1 hour
        for step in range(10):
            state = state_manager.get_state()
            photo_output = photosynthesis.forward(state, time=step*dt, dt=dt)
            
            # Track carbon flux
            gpp = _value(photo_output, 'gross_photosynthesis').item()
            respiration = _value(
                photo_output, 'diagnostics.photosynthesis.dark_respiration'
            ).item()
            net = _value(photo_output, 'net_photosynthesis').item()
            
            # Verify instant carbon balance
            assert abs((gpp - respiration) - net) < 1e-6, \
                f"Carbon imbalance at step {step}: GPP({gpp}) - Rd({respiration}) != Net({net})"
                
            carbon_fluxes.append({
                'gpp': gpp,
                'respiration': respiration,
                'net': net
            })
            
            state_manager.apply_updates(
                photo_output, dt=dt, check_conservation=False
            )
                    
        # Verify accumulated carbon makes sense
        total_net = sum(f['net'] for f in carbon_fluxes) * dt
        total_gpp = sum(f['gpp'] for f in carbon_fluxes) * dt
        total_resp = sum(f['respiration'] for f in carbon_fluxes) * dt
        
        assert abs((total_gpp - total_resp) - total_net) < 1e-3, \
            "Accumulated carbon imbalance"
        assert total_gpp >= 0, "Negative total GPP"
        assert total_resp >= 0, "Negative total respiration"


class TestParameterAuditIntegration:
    """Test parameter audit and validation through the system."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.param_manager = get_global_manager()
        self.param_manager.load_from_database()
        
    def test_parameter_audit_completeness(self):
        """Test that parameter audit finds all required parameters."""
        # Run parameter audit
        audit_report = self.param_manager.run_registry_audit()
        
        # Check key photosynthesis parameters are present
        required_params = [
            'biosphere.photosynthesis.vcmax25',
            'biosphere.photosynthesis.jmax25',
            'biosphere.photosynthesis.alpha',
            'biosphere.respiration.rd_q10',
            'validation.numerical.epsilon_divide',
            'validation.numerical.max_ppfd_validation'
        ]
        
        for param in required_params:
            spec = self.param_manager.get_parameter(param)
            assert spec is not None, f"Required parameter {param} not in registry"
            assert spec.value is not None, f"Parameter {param} has no value"
            assert spec.units is not None, f"Parameter {param} has no units"
            
    def test_missing_parameter_detection(self):
        """Test that missing parameters are properly detected and reported."""
        model = FarquharPhotosynthesis()

        with pytest.raises(ValueError, match="Required parameter"):
            model.get_parameter_value(
                "test_missing_param_detection",
                namespace="test.namespace",
                required=True,
            )

        assert self.param_manager.get_parameter(
            "test.namespace.test_missing_param_detection"
        ) is None

    def test_canonical_parameter_names_and_units(self):
        """Test canonical VPD/PPFD units and absence of retired parameters."""
        vpd_d0 = self.param_manager.get_parameter(
            "validation.numerical.vpd_response_d0"
        )
        max_ppfd = self.param_manager.get_parameter(
            "validation.numerical.max_ppfd_validation"
        )

        assert vpd_d0.units == "Pa"
        assert vpd_d0.value == pytest.approx(1500.0)
        assert max_ppfd.units == "umol photons m-2 s-1"
        assert self.param_manager.get_parameter(
            "validation.numerical.max_par_validation"
        ) is None
        for parameter_name in (
            "dummy_temperature",
            "dummy_co2",
            "dummy_ppfd",
            "dummy_par",
            "dummy_vcmax_scaling",
        ):
            assert self.param_manager.get_parameter(
                f"biosphere.photosynthesis.{parameter_name}"
            ) is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

