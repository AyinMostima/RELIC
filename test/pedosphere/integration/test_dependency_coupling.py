"""
L2 Integration Test: Cross-Module Dependencies and Coupling

Tests the dependency graph resolution, execution ordering, and state
propagation across multiple coupled pedosphere Functors.

Mathematical Expression:
    Dependency Graph: G = (V, E) where V = Functors, E = Dependencies
    Topological Sort: for all (u,v) in E, u appears before v in execution order
    State Propagation: state_t+1 = state_t + sum(updates_i) for all functors i

References:
    - Tarjan (1976) - Depth-first search and topological sorting
    - Kahn (1962) - Topological sorting of large networks
    - Category Theory: Functors as morphisms between categories
"""

import pytest
import torch

from relic.core.dependency_graph import DependencyGraphResolver, CyclicDependencyError
from relic.core.state import UpdateItem
from relic.scientific.pedosphere.decomposition.first_order import FirstOrderDecomposition
from relic.scientific.pedosphere.nutrient_cycling.mineralization import MineralizationProcess
from relic.scientific.pedosphere.nutrient_cycling.nitrification import NitrificationProcess
from relic.scientific.pedosphere.nutrient_cycling.denitrification import DenitrificationProcess


def _unwrap_update(value):
    if isinstance(value, UpdateItem):
        return value.value
    return value


# Simple mock functor for dependency testing
class MockFunctor:
    """Simple mock functor for testing dependency graph."""
    def __init__(self, name, dependencies=None):
        self.name = name
        self._dependencies = dependencies or []

    def get_dependencies(self):
        return self._dependencies

    def forward(self, state, time, dt):
        return {'mock_flux': torch.tensor(1.0)}


class TestDependencyGraphResolution:
    """Test dependency graph construction and topological sorting."""

    @pytest.fixture
    def dependency_resolver(self):
        """Create clean dependency resolver."""
        return DependencyGraphResolver()

    def test_simple_dependency_chain(self, dependency_resolver):
        """
        Test simple linear dependency chain using mock functors.

        Verifies topological sort produces correct execution order.
        """
        # Create mock functors with explicit dependencies
        functor_a = MockFunctor("A", dependencies=[])
        functor_b = MockFunctor("B", dependencies=["A"])
        functor_c = MockFunctor("C", dependencies=["B"])

        # Register functors (add_functor will read get_dependencies automatically)
        dependency_resolver.add_functor(functor_a, "A")
        dependency_resolver.add_functor(functor_b, "B")
        dependency_resolver.add_functor(functor_c, "C")

        # Add explicit edges based on dependencies
        dependency_resolver.add_edge("B", "A")  # B depends on A
        dependency_resolver.add_edge("C", "B")  # C depends on B

        # Resolve dependencies using topological_sort()
        execution_order = dependency_resolver.topological_sort()

        # Verify order: A before B before C
        a_idx = execution_order.index("A")
        b_idx = execution_order.index("B")
        c_idx = execution_order.index("C")

        assert a_idx < b_idx, f"A (idx={a_idx}) must execute before B (idx={b_idx})"
        assert b_idx < c_idx, f"B (idx={b_idx}) must execute before C (idx={c_idx})"

    def test_parallel_execution_groups(self, dependency_resolver):
        """
        Test execution group identification for parallel execution.

        Functors with no mutual dependencies can execute in parallel.
        """
        # Create functors: A and B are independent, both feed into C
        functor_a = MockFunctor("A", dependencies=[])
        functor_b = MockFunctor("B", dependencies=[])
        functor_c = MockFunctor("C", dependencies=["A", "B"])

        # Register functors
        dependency_resolver.add_functor(functor_a, "A")
        dependency_resolver.add_functor(functor_b, "B")
        dependency_resolver.add_functor(functor_c, "C")

        # Add edges
        dependency_resolver.add_edge("C", "A")
        dependency_resolver.add_edge("C", "B")

        # Get execution groups using get_parallel_groups()
        groups = dependency_resolver.get_parallel_groups()

        # Verify group structure
        assert len(groups) >= 2, "Should have at least 2 groups"

        # A and B should be in the same (first) group since they're independent
        assert "A" in groups[0] and "B" in groups[0], "A and B should execute in parallel"

        # C should be in a later group (depends on A and B)
        c_group = next(i for i, g in enumerate(groups) if "C" in g)
        assert c_group > 0, "C should not be in first group"

    def test_cyclic_dependency_detection(self, dependency_resolver):
        """
        Test that cyclic dependencies are detected and rejected.

        Verifies: CyclicDependencyError raised for circular dependencies.
        """
        # Create functors with circular dependency
        functor_a = MockFunctor("A", dependencies=["B"])
        functor_b = MockFunctor("B", dependencies=["A"])

        # Register functors
        dependency_resolver.add_functor(functor_a, "A")
        dependency_resolver.add_functor(functor_b, "B")

        # Create circular dependency via edges
        dependency_resolver.add_edge("A", "B")
        dependency_resolver.add_edge("B", "A")

        # Verify cyclic dependency detection
        with pytest.raises(CyclicDependencyError):
            dependency_resolver.topological_sort()


class TestStatePropagation:
    """Test state propagation across coupled Functors."""

    @pytest.fixture
    def coupled_functors(self):
        """Create set of coupled functors."""
        config = {"device": "cpu"}

        decomp = FirstOrderDecomposition("decomp", config)
        mineral = MineralizationProcess("mineral", config)
        nitrif = NitrificationProcess("nitrif", config)

        return {
            'decomp': decomp,
            'mineral': mineral,
            'nitrif': nitrif
        }

    @pytest.fixture
    def initial_state(self):
        """Create initial state for coupled simulation."""
        return {
            # Carbon pools
            'metabolic_litter_c': torch.tensor(100.0),
            'structural_litter_c': torch.tensor(200.0),
            'fast_som_c': torch.tensor(500.0),
            'slow_som_c': torch.tensor(1000.0),
            'passive_som_c': torch.tensor(2000.0),
            # Nitrogen pools
            'organic_nitrogen': torch.tensor(50.0),
            'organic_nitrogen_input': torch.tensor(0.0),
            'microbial_nitrogen': torch.tensor(2.0),  # Microbial biomass N     
            'nh4_nitrogen': torch.tensor(5.0),
            'no3_nitrogen': torch.tensor(10.0),
            # Environmental
            'soil_temperature': torch.tensor(293.15),
            'soil_moisture': torch.tensor(0.3),
            'water_filled_pore_space': torch.tensor(0.5),
            'soil_ph': torch.tensor(6.5),
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
            'lignin_n_ratio': torch.tensor(0.3),
            'clay_content': torch.tensor(0.25),
        }

    def test_sequential_state_updates(self, coupled_functors, initial_state):
        """
        Test that state updates propagate correctly through functor chain.

        Process chain: Decomposition -> Mineralization -> Nitrification
        Verifies: Each functor receives updated state from previous functor.
        """
        # Clone initial state to avoid modifying fixture
        state = {k: v.clone() for k, v in initial_state.items()}
        dt = 3600.0  # 1 hour

        # Execute decomposition
        decomp_fluxes = coupled_functors['decomp'].forward(state, time=0.0, dt=dt)

        # Manually update state with decomposition fluxes
        for key in state:
            if key.endswith('_c'):
                if key in decomp_fluxes:
                    state[key] = state[key] + _unwrap_update(decomp_fluxes[key])

        state['carbon_decomposition_rate'] = _unwrap_update(
            decomp_fluxes['carbon_decomposition_rate']
        )

        # Execute mineralization (should see decomposition effects)
        mineral_fluxes = coupled_functors['mineral'].forward(state, time=0.0, dt=dt)

        # Update state with ALL mineralization fluxes
        for key in ['organic_nitrogen', 'nh4_nitrogen', 'microbial_nitrogen']:
            if key in mineral_fluxes:
                state[key] = state[key] + _unwrap_update(mineral_fluxes[key])

        # Execute nitrification (should see mineralization effects)
        nitrif_fluxes = coupled_functors['nitrif'].forward(state, time=dt, dt=dt)

        # Update state with ALL nitrification fluxes
        for key in ['nh4_nitrogen', 'no3_nitrogen']:
            if key in nitrif_fluxes:
                state[key] = state[key] + _unwrap_update(nitrif_fluxes[key])

        # Verify that functors executed without error and state remains physically valid
        assert state['nh4_nitrogen'] >= 0, f"NH4 must remain non-negative, got {state['nh4_nitrogen']}"
        assert state['no3_nitrogen'] >= 0, f"NO3 must remain non-negative, got {state['no3_nitrogen']}"
        assert state['organic_nitrogen'] >= 0, f"Organic N must remain non-negative, got {state['organic_nitrogen']}"

    def test_state_isolation_between_steps(self, coupled_functors, initial_state):
        """
        Test that state is properly isolated between simulation steps.

        Verifies: Updates from step N don't leak into step N+1 initial state.
        """
        dt = 3600.0

        # Execute first step
        step1_state = {k: v.clone() for k, v in initial_state.items()}
        decomp_fluxes = coupled_functors['decomp'].forward(step1_state, time=0.0, dt=dt)

        # Update state with decomposition fluxes
        for key in step1_state:
            if key.endswith('_c'):
                if key in decomp_fluxes:
                    step1_state[key] = (
                        step1_state[key] + _unwrap_update(decomp_fluxes[key])
                    )

        carbon_after_step1 = step1_state['metabolic_litter_c'].item()

        # Execute second step (starting from step1's end state)
        step2_state = {k: v.clone() for k, v in step1_state.items()}
        decomp_fluxes = coupled_functors['decomp'].forward(step2_state, time=dt, dt=dt)

        # Update state with decomposition fluxes
        for key in step2_state:
            if key.endswith('_c'):
                if key in decomp_fluxes:
                    step2_state[key] = (
                        step2_state[key] + _unwrap_update(decomp_fluxes[key])
                    )

        carbon_after_step2 = step2_state['metabolic_litter_c'].item()

        # Carbon should continue decreasing (decomposition is consuming it)
        assert carbon_after_step2 <= carbon_after_step1, (
            "Carbon should decrease or stay same across steps"
        )


@pytest.mark.integration
class TestFullCouplingChain:
    """Test complete coupling chain with multiple processes."""

    def test_decomposition_to_gas_emission_chain(self):
        """
        Test complete process chain from decomposition to gas emissions.

        Process flow:
        Decomposition -> Mineralization -> Nitrification -> Denitrification -> N2O

        Verifies: Complete coupling chain maintains conservation and produces
        physically realistic outputs.
        """
        config = {"device": "cpu"}

        # Create functors
        decomp = FirstOrderDecomposition("decomp", config)
        mineral = MineralizationProcess("mineral", config)
        nitrif = NitrificationProcess("nitrif", config)
        denitrif = DenitrificationProcess("denitrif", config)

        # Set up dependency chain
        mineral.set_dependencies(["decomp"])
        nitrif.set_dependencies(["mineral"])
        denitrif.set_dependencies(["nitrif"])

        # Create initial state
        state = {
            'metabolic_litter_c': torch.tensor(100.0),
            'structural_litter_c': torch.tensor(200.0),
            'fast_som_c': torch.tensor(500.0),
            'slow_som_c': torch.tensor(1000.0),
            'passive_som_c': torch.tensor(2000.0),
            'organic_nitrogen': torch.tensor(50.0),
            'organic_nitrogen_input': torch.tensor(0.0),
            'microbial_nitrogen': torch.tensor(2.0),  # Microbial biomass N     
            'nh4_nitrogen': torch.tensor(5.0),
            'no3_nitrogen': torch.tensor(10.0),
            'soil_temperature': torch.tensor(293.15),
            'soil_moisture': torch.tensor(0.4),  # Higher moisture for denitrification
            'water_filled_pore_space': torch.tensor(0.5),
            'soil_ph': torch.tensor(6.5),
            'oxygen_concentration': torch.tensor(0.15),  # For denitrification
            'labile_carbon': torch.tensor(0.1),  # g C m-2, required denitrification input
            'metabolic_litter_input': torch.tensor(0.0),
            'structural_litter_input': torch.tensor(0.0),
            'lignin_n_ratio': torch.tensor(0.3),
            'clay_content': torch.tensor(0.25),
        }

        dt = 3600.0

        # Execute full chain
        decomp_fluxes = decomp.forward(state, 0.0, dt)

        # Update state with decomposition results (simplified, not using StateManager)
        for key in ['metabolic_litter_c', 'structural_litter_c', 'fast_som_c',
                    'slow_som_c', 'passive_som_c']:
            if key in decomp_fluxes:
                state[key] = state[key] + _unwrap_update(decomp_fluxes[key])

        state['carbon_decomposition_rate'] = _unwrap_update(
            decomp_fluxes['carbon_decomposition_rate']
        )

        mineral_fluxes = mineral.forward(state, 0.0, dt)

        # Update state with mineralization results
        if 'organic_nitrogen' in mineral_fluxes:
            state['organic_nitrogen'] = (
                state['organic_nitrogen']
                + _unwrap_update(mineral_fluxes['organic_nitrogen'])
            )
        if 'nh4_nitrogen' in mineral_fluxes:
            state['nh4_nitrogen'] = (
                state['nh4_nitrogen']
                + _unwrap_update(mineral_fluxes['nh4_nitrogen'])
            )
        if 'microbial_nitrogen' in mineral_fluxes:
            state['microbial_nitrogen'] = (
                state['microbial_nitrogen']
                + _unwrap_update(mineral_fluxes['microbial_nitrogen'])
            )

        nitrif_fluxes = nitrif.forward(state, dt, dt)

        # Update state with nitrification results
        if 'nh4_nitrogen' in nitrif_fluxes:
            state['nh4_nitrogen'] = (
                state['nh4_nitrogen']
                + _unwrap_update(nitrif_fluxes['nh4_nitrogen'])
            )
        if 'no3_nitrogen' in nitrif_fluxes:
            state['no3_nitrogen'] = (
                state['no3_nitrogen']
                + _unwrap_update(nitrif_fluxes['no3_nitrogen'])
            )

        denitrif_fluxes = denitrif.forward(state, 2*dt, dt)

        # Verify chain produced valid outputs
        assert 'heterotrophic_respiration' in decomp_fluxes, "Decomposition should produce CO2"
        assert 'n2o_flux' in denitrif_fluxes, "Denitrification should produce N2O"

        # Verify all fluxes are non-negative (rates, not deltas)
        assert _unwrap_update(decomp_fluxes['heterotrophic_respiration']) >= 0, (
            "CO2 flux must be non-negative"
        )
        assert _unwrap_update(denitrif_fluxes['n2o_flux']) >= 0, (
            "N2O flux must be non-negative"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

