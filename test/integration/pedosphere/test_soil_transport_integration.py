"""
Integration tests for soil transport processes.

Tests the combined functionality of solute advection-diffusion and gas diffusion
modules, ensuring they work together properly and maintain mass conservation.
"""

import math
import warnings

import pytest
import torch

from relic.core.dependency_graph import DependencyGraphResolver
from relic.core.simulator import RELICSimulator
from relic.scientific.pedosphere.transport.solute_advection_diffusion import SoluteAdvectionDiffusion1D
from relic.scientific.pedosphere.transport.gas_diffusion import GasDiffusion1D, DiffusivityModel
from relic.core.state import StateManager, UpdateItem


pytestmark = pytest.mark.usefixtures(
    "pedosphere_transport_constructor_overrides"
)


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


class TestSoilTransportIntegration:
    """Integration tests for coupled soil transport processes."""
    
    @pytest.fixture
    def comprehensive_state(self):
        """Create a comprehensive state for integration testing."""
        return {
            # Soil physics
            'soil_water_content': torch.tensor([0.25, 0.22, 0.20, 0.18, 0.15]),
            'soil_moisture': torch.tensor([0.25, 0.22, 0.20, 0.18, 0.15]),
            'porosity': torch.tensor([0.45, 0.43, 0.41, 0.39, 0.37]),
            'bulk_density': torch.tensor([1200, 1250, 1300, 1350, 1400]),  # kg m-3
            'clay_content': torch.tensor([0.15, 0.18, 0.20, 0.22, 0.25]),
            'soil_temperature': torch.tensor([288.15, 287.15, 286.15, 285.15, 284.15]),
            
            # Hydrology  
            'darcy_flux': torch.tensor([1e-6, 8e-7, 6e-7, 4e-7, 2e-7]),  # m/s
            
            # Solute concentrations
            'no3_concentration': torch.tensor([5.0, 4.5, 4.0, 3.5, 3.0]),  # g N m-3
            'doc_concentration': torch.tensor([50.0, 45.0, 40.0, 35.0, 30.0]),  # g C m-3
            'surface_no3_concentration': torch.tensor(5.0),
            'surface_doc_concentration': torch.tensor(50.0),
            
            # Gas concentrations
            'soil_co2_concentration': torch.tensor([3.0, 2.7, 2.4, 2.1, 1.8]),  # g C m-3
            'n2o_concentration': torch.tensor([0.005, 0.004, 0.003, 0.002, 0.001]),  # g N m-3
            
            # Pools for mass conservation
            'soil_carbon': torch.tensor(5000.0),  # g C m-2
            'NO3': torch.tensor(50.0),            # g N m-2
            'doc': torch.tensor(100.0),           # g C m-2
        }
    
    @pytest.fixture
    def solute_functor(self):
        """Create solute advection-diffusion functor."""
        return SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            cfl_number=0.5,
            max_timestep=600.0
        )
    
    @pytest.fixture 
    def gas_functor(self):
        """Create gas diffusion functor."""
        return GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            diffusivity_model=DiffusivityModel.MILLINGTON_QUIRK
        )
    
    def test_coupled_execution(self, comprehensive_state, solute_functor, gas_functor):
        """Test coupled execution of solute and gas transport."""
        dt = 300.0  # 5 minutes
        time = 0.0
        
        # Execute solute transport
        solute_output = solute_functor.forward(comprehensive_state, time, dt)
        
        # Execute gas transport
        gas_output = gas_functor.forward(comprehensive_state, time, dt)
        
        # Check basic outputs exist
        assert 'nitrogen_leaching_flux' in solute_output
        assert 'carbon_leaching_flux' in solute_output
        assert 'co2_surface_flux' in gas_output
        assert 'n2o_surface_flux' in gas_output
        assert 'heterotrophic_respiration' not in gas_output
        assert 'n2o_flux' not in gas_output
        
        # Check physical constraints
        assert torch.all(_unwrap_update(solute_output['nitrogen_leaching_flux']) >= 0)
        assert torch.all(_unwrap_update(solute_output['carbon_leaching_flux']) >= 0)
        assert torch.all(torch.isfinite(_unwrap_update(gas_output['co2_surface_flux'])))
        assert torch.all(torch.isfinite(_unwrap_update(gas_output['n2o_surface_flux'])))
    
    def test_mass_conservation_integration(self, comprehensive_state, solute_functor):
        """Test mass conservation in integrated transport system."""
        dt = 300.0
        time = 0.0
        
        # Execute the canonical finite-volume transport ledger.
        output = solute_functor.forward(comprehensive_state, time, dt)

        audit = solute_functor.check_conservation_laws(
            comprehensive_state,
            output,
            dt,
        )
        assert audit["mass"] < 1.0e-6
        assert abs(float(output[
            "diagnostics.solute_transport.nitrogen_mass_balance_residual"
        ].value)) < 1.0e-6
        assert abs(float(output[
            "diagnostics.solute_transport.carbon_mass_balance_residual"
        ].value)) < 1.0e-6
    
    def test_multi_timestep_stability(self, comprehensive_state, solute_functor, gas_functor):
        """Test stability over multiple time steps."""
        dt = 300.0
        n_steps = 10
        
        state = comprehensive_state.copy()
        
        for step in range(n_steps):
            time = step * dt
            
            # Execute both transport processes
            solute_output = solute_functor.forward(state, time, dt)
            gas_output = gas_functor.forward(state, time, dt)
            
            # Check for numerical stability (no NaN/Inf)
            for key, value in solute_output.items():
                value_tensor = _unwrap_update(value)
                if torch.is_tensor(value_tensor):
                    assert torch.isfinite(value_tensor).all(), f"Non-finite values in {key} at step {step}"
            
            for key, value in gas_output.items():
                value_tensor = _unwrap_update(value)
                if torch.is_tensor(value_tensor):
                    assert torch.isfinite(value_tensor).all(), f"Non-finite values in {key} at step {step}"
            
            # Check physical bounds
            if 'no3_concentration' in solute_output:
                no3_updated = _unwrap_update(solute_output['no3_concentration'])
                assert torch.all(no3_updated >= 0)
            if 'doc_concentration' in solute_output:
                doc_updated = _unwrap_update(solute_output['doc_concentration'])
                assert torch.all(doc_updated >= 0)

            state['no3_concentration'] = _unwrap_update(
                solute_output['no3_concentration']
            )
            state['doc_concentration'] = _unwrap_update(
                solute_output['doc_concentration']
            )
    
    def test_parameter_sensitivity(self, comprehensive_state):
        """Test sensitivity to key parameters."""
        dt = 300.0
        time = 0.0
        
        # Test different CFL numbers
        cfl_values = [0.1, 0.5, 0.8]
        results = {}
        
        for cfl in cfl_values:
            functor = SoluteAdvectionDiffusion1D(
                n_layers=5,
                soil_depth=1.0,
                cfl_number=cfl
            )
            
            output = functor.forward(comprehensive_state, time, dt)
            results[cfl] = _unwrap_update(output['nitrogen_leaching_flux'])
        
        # Results should vary with CFL number
        # (though they might be similar for stable cases)
        assert len(results) == len(cfl_values)
        
        # All results should be physically reasonable
        for cfl, flux in results.items():
            assert torch.all(flux >= 0), f"Negative flux for CFL={cfl}"
            assert torch.isfinite(flux).all(), f"Non-finite flux for CFL={cfl}"

        torch.testing.assert_close(results[0.1], results[0.5], rtol=5.0e-3, atol=1.0e-12)
        torch.testing.assert_close(results[0.5], results[0.8], rtol=5.0e-3, atol=1.0e-12)
    
    def test_different_diffusivity_models(self, comprehensive_state):
        """Test different gas diffusivity models."""
        dt = 300.0
        time = 0.0
        
        models = [
            DiffusivityModel.MILLINGTON_QUIRK,
            DiffusivityModel.PENMAN,
            DiffusivityModel.DNDC
        ]
        
        results = {}
        
        for model in models:
            functor = GasDiffusion1D(
                n_layers=5,
                soil_depth=1.0,
                diffusivity_model=model
            )
            
            output = functor.forward(comprehensive_state, time, dt)
            results[model.value] = _unwrap_update(output['co2_surface_flux'])
        
        # Surface flux sign follows the concentration-gradient direction.
        for model_name, flux in results.items():
            assert torch.isfinite(flux).all(), f"Non-finite flux for {model_name}"
            assert torch.abs(flux) > 0, f"Zero flux for {model_name}"
        assert len({float(flux) for flux in results.values()}) > 1
    
    def test_temperature_effects(self, comprehensive_state):
        """Test temperature effects on transport processes."""
        dt = 300.0
        time = 0.0
        
        # Test at different temperatures
        temperatures = [278.15, 288.15, 298.15]  # 5 C, 15 C, 25 C
        
        gas_functor = GasDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            enable_temperature_correction=True
        )
        
        co2_fluxes = []
        
        for temp in temperatures:
            state = comprehensive_state.copy()
            state['soil_temperature'] = torch.full((5,), temp)

            output = gas_functor.forward(state, time, dt)
            co2_fluxes.append(float(_unwrap_update(output['co2_surface_flux'])))
        
        # Higher temperatures should change diffusion magnitude.
        assert all(math.isfinite(flux) for flux in co2_fluxes)
        assert len(set(co2_fluxes)) > 1  # Should vary with temperature
    
    def test_moisture_effects(self, comprehensive_state):
        """Test moisture effects on transport processes."""
        dt = 300.0
        time = 0.0
        
        # Test at different moisture levels
        moisture_levels = [0.1, 0.2, 0.3]
        
        gas_functor = GasDiffusion1D(n_layers=5, soil_depth=1.0)
        
        co2_fluxes = []
        
        for moisture in moisture_levels:
            state = comprehensive_state.copy()
            state['soil_water_content'] = torch.full((5,), moisture)
            state['soil_moisture'] = torch.full((5,), moisture)

            output = gas_functor.forward(state, time, dt)
            co2_fluxes.append(float(_unwrap_update(output['co2_surface_flux'])))
        
        assert all(math.isfinite(flux) for flux in co2_fluxes)
        
        # Generally expect lower CO2 flux at higher moisture
        # (due to reduced air-filled porosity)
        assert abs(co2_fluxes[0]) > abs(co2_fluxes[-1])

    def test_solute_simulator_replaces_full_state_and_records_provenance(
        self,
        comprehensive_state,
    ):
        """The real StateManager applies one full-state solute transaction."""

        dt = 300.0
        state = {
            key: value.clone()
            for key, value in comprehensive_state.items()
            if key
            in {
                "soil_water_content",
                "darcy_flux",
                "no3_concentration",
                "doc_concentration",
                "bulk_density",
                "porosity",
                "soil_temperature",
                "surface_no3_concentration",
                "surface_doc_concentration",
            }
        }
        direct_functor = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
            cfl_number=0.5,
            max_timestep=600.0,
        )
        direct = direct_functor.forward(
            {key: value.clone() for key, value in state.items()},
            time=0.0,
            dt=dt,
        )

        state_manager = StateManager(
            device="cpu",
            conservation_checks={"enabled": True, "raise_on_violation": True},
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
                "conservation_checks": True,
                "strict_dependencies": False,
                "strict_scale_detection": False,
            }
        )
        simulator.initialize(state_manager, DependencyGraphResolver())
        simulator.register_functor(
            SoluteAdvectionDiffusion1D(
                n_layers=5,
                soil_depth=1.0,
                cfl_number=0.5,
                max_timestep=600.0,
            ),
            execution_group="fast",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            step_updates = simulator.step(dt=dt)

        assert set(step_updates) == set(direct_functor.metadata.output_variables)
        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, UserWarning)
        ]
        for key in ("no3_concentration", "doc_concentration"):
            torch.testing.assert_close(
                state_manager.get_state(key),
                direct[key].value,
                rtol=0.0,
                atol=1.0e-12,
            )
            writers = {
                record["functor"]
                for record in simulator.get_variable_provenance(key)
                if record.get("role") in {"increment_source", "final_writer"}
            }
            assert writers == {"SoluteAdvectionDiffusion1D"}

        for residual in (
            "diagnostics.solute_transport.nitrogen_mass_balance_residual",
            "diagnostics.solute_transport.carbon_mass_balance_residual",
        ):
            assert abs(float(direct[residual].value)) <= 1.0e-10
        assert str(simulator.get_effective_device("SoluteAdvectionDiffusion1D")) == "cpu"


class TestSoilTransportBenchmark:
    """Benchmark tests against analytical solutions where available."""
    
    def test_pure_diffusion_analytical(self):
        """Test pure diffusion against analytical solution."""
        # Simple 1D diffusion with constant D
        dt = 60.0
        time = 0.0
        
        # Create simple uniform conditions
        state = {
            'soil_water_content': torch.full((5,), 0.3),
            'porosity': torch.full((5,), 0.4),
            'bulk_density': torch.full((5,), 1300.0),
            'clay_content': torch.full((5,), 0.2),
            'soil_temperature': torch.full((5,), 288.15),
            'darcy_flux': torch.zeros(5),  # Pure diffusion
            'no3_concentration': torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0]),
            'doc_concentration': torch.tensor([50.0, 40.0, 30.0, 20.0, 10.0]),
            'surface_no3_concentration': torch.tensor(10.0),
            'surface_doc_concentration': torch.tensor(50.0),
        }
        
        functor = SoluteAdvectionDiffusion1D(
            n_layers=5,
            soil_depth=1.0,
        )
        
        output = functor.forward(state, time, dt)
        
        # Check that diffusion moves material from high to low concentration
        # This is a qualitative check rather than exact analytical comparison
        assert torch.isfinite(_unwrap_update(output['nitrogen_leaching_flux'])).all()
        assert torch.isfinite(_unwrap_update(output['carbon_leaching_flux'])).all()
        assert _unwrap_update(output['nitrogen_leaching_flux']) == 0.0
        assert _unwrap_update(output['carbon_leaching_flux']) == 0.0
        no3_final = _unwrap_update(output['no3_concentration'])
        total_variation_before = torch.sum(
            torch.abs(state['no3_concentration'][1:] - state['no3_concentration'][:-1])
        )
        total_variation_after = torch.sum(torch.abs(no3_final[1:] - no3_final[:-1]))
        assert total_variation_after < total_variation_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

