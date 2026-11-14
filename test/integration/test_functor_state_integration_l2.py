#!/usr/bin/env python
"""
L2 Integration Tests for Functor-State-Simulator Integration

Tests the integration between:
1. Functors returning increments (not modifying state)
2. StateManager applying updates transactionally
3. Simulator orchestrating execution
4. Conservation validation during updates

Mathematical Tests:
1. Mass conservation across updates
2. Energy conservation with tolerances
3. Multi-step temporal integration
4. Dependency resolution and parallel execution
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from relic.core.functor import FunctorMetadata, ProcessCategory
from relic.core.state import StateManager, UpdateItem, UpdateMode
from relic.core.state.types import StateVariableSpec
from relic.core.simulator import RELICSimulator
from relic.core.functor import BiologicalProcessFunctor, TransportMechanismFunctor


# Units this test declares for its own uncontracted state variables. The
# simulator infers no units from variable names (relic/core/simulator/state_init.py);
# the test is the caller that declares them through the state_units setting.
_TEST_STATE_UNITS = {
    'carbon_atmosphere': 'kg C m-2',
    'carbon_pool': 'kg C m-2',
    'co2': 'dimensionless',
    'soil_water': 'dimensionless',
}


def _declared_simulator(config=None, *args, **kwargs):
    """Construct RELICSimulator with this test's state_units declaration."""
    declared = dict(config or {})
    declared.setdefault("state_units", _TEST_STATE_UNITS)
    return RELICSimulator(declared, *args, **kwargs)


def _test_metadata(name, category, inputs, outputs, optional_outputs=()):
    """Build complete metadata for integration test Functors."""
    return FunctorMetadata(
        name=name,
        description=f"Integration test fixture for {name}",
        category=category,
        references=["RELIC integration test fixture"],
        input_variables=list(inputs),
        output_variables=list(outputs),
        optional_output_variables=list(optional_outputs),
    )


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _total_carbon(state):
    return _to_float(state["carbon_pool"]) + _to_float(state["carbon_atmosphere"])


def _stub_q10_response(temperature):
    """Temperature law of the test fixtures below (a concrete Functor owns its law;
    the bridge provides none).

    References: van 't Hoff (1884) Q10 form, as used by the retired bridge default.
    Mathematical expression: f(T) = Q10 ** ((T - T_ref) / 10).
    Parameter meanings: T temperature; Q10 = 2 the rate ratio per 10 K;
        T_ref = 298.15 K the reference temperature.
    Units: T and T_ref in K; f dimensionless.
    Domain/constraints: fixture law for any finite T; f(T_ref) = 1.
    Sign convention: none (a positive multiplier on a rate).
    """
    return 2.0 ** ((temperature - 298.15) / 10.0)


class PhotosynthesisFunctor(BiologicalProcessFunctor):
    """Test photosynthesis functor producing carbon."""

    optional_input_variables = ("carbon_atmosphere",)
    
    def __init__(self, name="photosynthesis", **kwargs):
        metadata = kwargs.pop(
            "metadata",
            _test_metadata(
                name,
                ProcessCategory.BIOLOGICAL_PROCESS,
                ("par", "co2", "temperature", "carbon_atmosphere"),
                ("carbon_pool", "carbon_atmosphere", "diagnostics.gpp_rate"),
                ("carbon_atmosphere",),
            ),
        )
        # Test-law coefficients come from the instance config (no caller default).
        config = kwargs.pop(
            "config",
            {"parameters": {"par_half_saturation": 500.0, "max_photosynthesis_rate": 1.0e-6}},
        )
        super().__init__(name=name, config=config, metadata=metadata, **kwargs)

    def temperature_function(self, temperature):
        return _stub_q10_response(temperature)

    def forward(self, state, time, dt, **kwargs):
        """Photosynthesis produces carbon."""
        # Simplified Farquhar model
        par = state.get("par", torch.tensor(1000.0))  # umol m-2 s-1
        co2 = state.get("co2", torch.tensor(400.0))  # ppm
        temp = state.get("temperature", torch.tensor(298.15))  # K
        
        # Apply temperature response
        temp_factor = self.temperature_function(temp)
        
        # Michaelis-Menten kinetics for light
        par_half = self.get_parameter_value("par_half_saturation")
        light_factor = par / (par + par_half)
        
        # CO2 response
        co2_ref = self.get_parameter_value("atmosphere.co2_reference", namespace="relic.physical.constants")
        co2_factor = co2 / co2_ref
        
        # GPP rate (kg C m-2 s-1)
        max_rate = self.get_parameter_value("max_photosynthesis_rate")
        gpp = max_rate * light_factor * co2_factor * temp_factor

        updates = {
            "carbon_pool": UpdateItem(
                value=gpp * dt,
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="kg C m-2",
                element="C",
                source_functor=self.name,
            ),
            "diagnostics.gpp_rate": gpp,
        }
        if "carbon_atmosphere" in state:
            updates["carbon_atmosphere"] = UpdateItem(
                value=-gpp * dt,
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="kg C m-2",
                element="C",
                source_functor=self.name,
            )
        return updates


class RespirationFunctor(BiologicalProcessFunctor):
    """Test respiration functor consuming carbon."""

    optional_input_variables = ("carbon_atmosphere",)
    
    def __init__(self, name="respiration", **kwargs):
        metadata = kwargs.pop(
            "metadata",
            _test_metadata(
                name,
                ProcessCategory.BIOLOGICAL_PROCESS,
                ("temperature", "carbon_pool", "carbon_atmosphere"),
                (
                    "carbon_pool",
                    "carbon_atmosphere",
                    "diagnostics.respiration_rate",
                ),
                ("carbon_atmosphere",),
            ),
        )
        config = kwargs.pop("config", {"parameters": {"base_respiration_rate": 5.0e-7}})
        super().__init__(name=name, config=config, metadata=metadata, **kwargs)

    def temperature_function(self, temperature):
        return _stub_q10_response(temperature)

    def forward(self, state, time, dt, **kwargs):
        """Respiration consumes carbon."""
        carbon = state.get("carbon_pool", torch.tensor(10.0))  # kg C m-2
        temp = state.get("temperature", torch.tensor(298.15))
        
        # Temperature response
        temp_factor = self.temperature_function(temp)
        
        # Base respiration rate
        base_rate = self.get_parameter_value("base_respiration_rate")
        
        # Respiration proportional to carbon pool
        respiration = base_rate * carbon * temp_factor

        updates = {
            "carbon_pool": UpdateItem(
                value=-respiration * dt,
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="kg C m-2",
                element="C",
                source_functor=self.name,
            ),
            "diagnostics.respiration_rate": respiration,
        }
        if "carbon_atmosphere" in state:
            updates["carbon_atmosphere"] = UpdateItem(
                value=respiration * dt,
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="kg C m-2",
                element="C",
                source_functor=self.name,
            )
        return updates


class WaterTransportFunctor(TransportMechanismFunctor):
    """Test water transport functor."""

    TRANSPORT_TYPE = "diffusion"
    
    def __init__(self, name="water_transport", **kwargs):
        metadata = kwargs.pop(
            "metadata",
            _test_metadata(
                name,
                ProcessCategory.TRANSPORT_MECHANISM,
                ("soil_water",),
                ("water_flux", "soil_water"),
            ),
        )
        config = kwargs.pop("config", {"parameters": {"drainage_rate": 0.01}})
        super().__init__(name=name, config=config, metadata=metadata, **kwargs)
        
    def forward(self, state, time, dt, **kwargs):
        """Water transport between compartments."""
        soil_water = state.get("soil_water", torch.tensor(100.0))  # mm
        
        # Simple drainage model
        drainage_rate = self.get_parameter_value("drainage_rate")  # fraction/s
        drainage = soil_water * drainage_rate
        
        # Convert to increments
        updates = self.flux_to_updates(drainage, dt, "water_flux")
        updates["soil_water"] = -drainage * dt  # Remove from soil
        
        return updates


def _float64_state(values: dict[str, float]) -> dict[str, torch.Tensor]:
    """Build a float64 initial state.

    The ledger audits the realized state deltas; a 1000 kg C m-2 pool receiving 0.0024 kg
    increments loses about 2e-5 kg per step in float32 storage, so these budgets need
    float64 storage to close within the tolerance they assert.
    """
    return {name: torch.tensor(float(value), dtype=torch.float64) for name, value in values.items()}


class TestFunctorStateIntegration:
    """Test functor-state integration."""
    
    def test_single_functor_update(self):
        """Test single functor updating state."""
        # Create state manager
        state_mgr = StateManager()
        state_mgr.initialize_state(_float64_state({
            "par": 1000.0,
            "co2": 400.0,
            "temperature": 298.15,
            "carbon_pool": 0.0,
            "carbon_atmosphere": 1000.0,
        }))
        
        # Create functor
        photo = PhotosynthesisFunctor()
        
        # Run functor
        state = state_mgr.get_state()
        updates = photo(state, time=0.0, dt=3600.0)  # 1 hour
        
        # Apply updates
        state_mgr.apply_updates(updates, dt=3600.0)
        
        # Check state was updated
        new_state = state_mgr.get_state()
        assert new_state["carbon_pool"] > 0.0
        
    def test_conservation_with_paired_functors(self):
        """Test carbon conservation with photosynthesis and respiration."""
        # Create state manager with conservation tracking
        state_mgr = StateManager()
        state_mgr.initialize_state(_float64_state({
            "par": 1000.0,
            "co2": 400.0,
            "temperature": 298.15,
            "carbon_pool": 10.0,
            "carbon_atmosphere": 1000.0
        }))
        
        # Set up carbon ledger
        state_mgr.set_conservation_ledger("carbon", ["carbon_pool", "carbon_atmosphere"])
        initial_carbon = _total_carbon(state_mgr.get_state())
        
        # Create functors
        photo = PhotosynthesisFunctor(
            config={"parameters": {"par_half_saturation": 500.0, "max_photosynthesis_rate": 1e-6}}
        )
        resp = RespirationFunctor(config={"base_respiration_rate": 5e-7})
        
        # Run both functors
        state = state_mgr.get_state()
        photo_updates = photo(state, time=0.0, dt=3600.0)
        resp_updates = resp(state, time=0.0, dt=3600.0)
        
        # Commit each closed transfer with its own explicit unit ledger.
        state_mgr.apply_updates(photo_updates, dt=3600.0)
        state_mgr.apply_updates(resp_updates, dt=3600.0)
        
        # Check conservation
        final_carbon = _total_carbon(state_mgr.get_state())
        carbon_change = abs(final_carbon - initial_carbon)
        
        # Should be conserved within StateManager dynamic tolerance
        assert carbon_change < state_mgr.conservation_tolerance * 3600.0
        
    def test_multi_step_integration(self):
        """Test multiple time steps with state evolution."""
        # Create state manager
        state_mgr = StateManager()
        state_mgr.initialize_state(_float64_state({
            "par": 1000.0,
            "co2": 400.0,
            "temperature": 298.15,
            "carbon_pool": 1.0,
            "carbon_atmosphere": 1000.0
        }))
        state_mgr.set_conservation_ledger("carbon", ["carbon_pool", "carbon_atmosphere"])
        
        # Create functors
        photo = PhotosynthesisFunctor(
            config={"parameters": {"par_half_saturation": 500.0, "max_photosynthesis_rate": 1e-5}}
        )
        resp = RespirationFunctor(config={"base_respiration_rate": 5e-7})
        
        # Run multiple time steps
        dt = 3600.0  # 1 hour
        n_steps = 24  # 24 hours
        
        carbon_history = []
        
        for step in range(n_steps):
            # Vary PAR over day (simplified diurnal cycle)
            par = 1000.0 * max(0, torch.sin(torch.tensor(step * 3.14159 / 24)))
            state_mgr.set_state("par", par, validate=False)
            
            # Get current state
            state = state_mgr.get_state()
            carbon_history.append(float(state["carbon_pool"]))
            
            # Run functors
            photo_updates = photo(state, time=step*dt, dt=dt)
            resp_updates = resp(state, time=step*dt, dt=dt)
            
            # Commit each closed transfer with its own explicit unit ledger.
            state_mgr.apply_updates(photo_updates, dt=dt)
            state_mgr.apply_updates(resp_updates, dt=dt)
        
        # Carbon should increase during day, decrease at night
        assert max(carbon_history) > carbon_history[0]
        assert min(carbon_history) < carbon_history[0]
        
    def test_conservation_violation_detection(self):
        """Test that conservation violations are detected."""
        # Create state manager with strict conservation
        state_mgr = StateManager()
        state_mgr.initialize_state({
            "carbon_pool": 100.0,
            "carbon_atmosphere": 1000.0
        })
        
        # Set up carbon ledger with strict checking
        state_mgr.set_conservation_ledger("carbon", ["carbon_pool", "carbon_atmosphere"])
        
        # Create invalid updates (carbon appears from nowhere)
        bad_updates = {
            "carbon_pool": UpdateItem(
                value=torch.tensor(50.0),
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
                unit="kg C m-2",
                element="C",
                source_functor="invalid_carbon_source",
            ),
            # Not removing from atmosphere - violation!
        }
        
        # Apply with strict conservation should raise error
        with pytest.raises(ValueError, match="Conservation violation"):
            state_mgr.apply_updates(
                bad_updates, 
                dt=1.0,
                check_conservation=True
            )


class TestSimulatorIntegration:
    """Test full simulator integration."""
    
    def test_simulator_with_functors(self):
        """Test simulator running functors."""
        # Create simulator
        config = {
            "time_step": 3600.0,  # 1 hour
            "max_time": 86400.0,   # 24 hours
            "conservation_checks": False
        }
        simulator = _declared_simulator(config)
        
        # Initialize state
        simulator.initialize_state({
            "par": 1000.0,
            "co2": 400.0,
            "temperature": 298.15,
            "carbon_pool": 10.0,
            "carbon_atmosphere": 1000.0,
            "soil_water": 100.0
        })
        # The test declares the carbon ledger membership its functors report
        # (UpdateItem element 'carbon'); membership is not inferred from names.
        for name in ("carbon_pool", "carbon_atmosphere"):
            simulator.state_manager.register_variable(
                StateVariableSpec(name=name, units="kg C m-2", conserved=True, accumulates="carbon.total")
            )
        
        # Add functors
        simulator.register_functor(PhotosynthesisFunctor())
        simulator.register_functor(RespirationFunctor())
        simulator.register_functor(WaterTransportFunctor())

        initial_state = simulator.state_manager.get_state()
        initial_carbon = _to_float(initial_state["carbon_pool"])
        initial_water = _to_float(initial_state["soil_water"])

        # Run simulation
        results = simulator.run()

        # Check results
        assert len(results) > 0
        assert "carbon_pool" in results
        assert "soil_water" in results

        # Carbon and water should change over time
        final_carbon = _to_float(results["carbon_pool"])
        final_water = _to_float(results["soil_water"])
        assert final_carbon != initial_carbon
        assert final_water != initial_water
        
    def test_dependency_resolution(self):
        """Test that dependencies are resolved correctly."""
        
        class FunctorA(BiologicalProcessFunctor):
            def __init__(self, name):
                super().__init__(
                    name=name,
                    metadata=_test_metadata(
                        name,
                        ProcessCategory.BIOLOGICAL_PROCESS,
                        (),
                        ("value_alpha",),
                    ),
                    config={},
                )
            def get_dependencies(self):
                return []
            def forward(self, state, time, dt, **kwargs):
                return {"value_alpha": torch.tensor(1.0)}
                
        class FunctorB(BiologicalProcessFunctor):
            def __init__(self, name):
                super().__init__(
                    name=name,
                    metadata=_test_metadata(
                        name,
                        ProcessCategory.BIOLOGICAL_PROCESS,
                        ("value_alpha",),
                        ("value_beta",),
                    ),
                    config={},
                )
            def get_dependencies(self):
                return ["functor_a"]
            def forward(self, state, time, dt, **kwargs):
                # Depends on A's output
                a_val = state.get("value_alpha", torch.tensor(0.0))
                return {"value_beta": a_val * 2.0}
                
        class FunctorC(BiologicalProcessFunctor):
            def __init__(self, name):
                super().__init__(
                    name=name,
                    metadata=_test_metadata(
                        name,
                        ProcessCategory.BIOLOGICAL_PROCESS,
                        ("value_beta",),
                        ("value_gamma",),
                    ),
                    config={},
                )
            def get_dependencies(self):
                return ["functor_b"]
            def forward(self, state, time, dt, **kwargs):
                # Depends on B's output
                b_val = state.get("value_beta", torch.tensor(0.0))
                return {"value_gamma": b_val * 3.0}
        
        # Create simulator
        simulator = _declared_simulator({"time_step": 1.0, "max_time": 1.0, "conservation_checks": False})
        simulator.initialize_state({})
        
        # Add functors in dependency order for simulator registration
        functor_a = FunctorA(name="functor_a")
        functor_b = FunctorB(name="functor_b")
        functor_c = FunctorC(name="functor_c")
        
        simulator.register_functor(functor_a)
        simulator.register_functor(functor_b)
        simulator.register_functor(functor_c)
        
        # Run one step
        updates = simulator.step(1.0)
        
        # Check execution order was correct
        # A should produce 1, B should produce 2, C should produce 6
        assert "value_alpha" in updates
        assert "value_beta" in updates
        assert "value_gamma" in updates
        value_gamma = updates["value_gamma"]
        if hasattr(value_gamma, "value"):
            value_gamma = value_gamma.value
        value_gamma = torch.as_tensor(value_gamma)
        assert torch.allclose(value_gamma, torch.tensor(6.0, dtype=value_gamma.dtype))
        
    def test_parallel_execution_groups(self):
        """Test that independent functors run in parallel groups."""
        
        class IndependentFunctor(BiologicalProcessFunctor):
            def __init__(self, name, value):
                super().__init__(
                    name=name,
                    metadata=_test_metadata(
                        name,
                        ProcessCategory.BIOLOGICAL_PROCESS,
                        (),
                        (f"{name}_output",),
                    ),
                    config={},
                )
                self.value = value
                
            def get_dependencies(self):
                return []
                
            def forward(self, state, time, dt, **kwargs):
                return {f"{self.name}_output": torch.tensor(self.value)}
        
        # Create simulator
        simulator = _declared_simulator({"time_step": 1.0, "max_time": 1.0, "conservation_checks": False})
        simulator.initialize_state({})
        
        # Add multiple independent functors
        for i in range(5):
            simulator.register_functor(IndependentFunctor(f"functor_{i}", float(i)))
        
        # Analyze execution groups
        from relic.core.dependency_graph import DependencyGraphResolver
        resolver = DependencyGraphResolver()
        
        # Build graph
        for functor in simulator.functors.values():
            resolver.add_functor(functor)
            
        groups = resolver.get_parallel_groups()
        
        # All independent functors should be in the same group
        assert len(groups) == 1
        assert len(groups[0]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

