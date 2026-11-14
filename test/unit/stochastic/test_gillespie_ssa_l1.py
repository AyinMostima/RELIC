"""
Unit Tests for Gillespie Stochastic Simulation Algorithm (L1)

This test suite validates the GillespieSSA implementation for exact stochastic
simulation of coupled biochemical reactions. Tests cover:
- Basic algorithm correctness
- Propensity function calculations
- Mass action kinetics
- Reaction system management
- Statistical properties
- Edge cases and error handling

Mathematical Validation:
    - Reaction propensities: a_j(x) = c_j * product_i binom(x_i, nu_ij^-)
    - Waiting time: tau ~ Exp(lambda) where lambda = sum(alpha_k)
    - Reaction selection: P(reaction k) = alpha_k / lambda
    - Stoichiometric updates: x_new = x_old + nu_k

References:
    - Gillespie, D.T. (1976). J. Comput. Phys. 22, 403-434
    - Gillespie, D.T. (1977). J. Phys. Chem. 81, 2340-2361
    - Gibson & Bruck (2000). J. Phys. Chem. A 104, 1876-1889

Test Categories:
    - L1.1: System Initialization and Validation
    - L1.2: Propensity Calculations
    - L1.3: Trajectory Generation
    - L1.4: Statistical Properties
    - L1.5: Reaction Management
    - L1.6: Edge Cases

Authors: Yuchong Long
"""

import pytest
import torch

# Import Gillespie SSA components
from relic.stochastic.gillespie import (
    GillespieSSA,
    Reaction,
    ReactionSystem
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_birth_death_system():
    """
    Simple birth-death process for testing.
    
    Reactions:
        Birth: empty_set -> S, rate = k_birth
        Death: S -> empty_set, rate = k_death
    
    Analytical steady-state mean: k_birth / k_death
    """
    species = ['S']
    reactions = [
        Reaction(
            name='birth',
            reactants={},  # Birth from nothing
            products={'S': 1},
            rate_constant=5.0  # births per unit time
        ),
        Reaction(
            name='death',
            reactants={'S': 1},
            products={},  # Death to nothing
            rate_constant=1.0  # deaths per molecule per unit time
        )
    ]
    initial_state = {'S': 10}
    
    system = ReactionSystem(
        species=species,
        reactions=reactions,
        initial_state=initial_state,
    )
    
    return system


@pytest.fixture
def enzymatic_reaction_system():
    """
    Michaelis-Menten enzymatic reaction.
    
    Reactions:
        E + S <-> ES (reversible binding)
        ES -> E + P (product formation)
    
    This tests second-order reactions and intermediate species.
    """
    species = ['E', 'S', 'ES', 'P']
    reactions = [
        Reaction(
            name='binding',
            reactants={'E': 1, 'S': 1},
            products={'ES': 1},
            rate_constant=0.01  # Binding rate
        ),
        Reaction(
            name='unbinding',
            reactants={'ES': 1},
            products={'E': 1, 'S': 1},
            rate_constant=0.1  # Unbinding rate
        ),
        Reaction(
            name='catalysis',
            reactants={'ES': 1},
            products={'E': 1, 'P': 1},
            rate_constant=0.05  # Catalytic rate
        )
    ]
    initial_state = {
        'E': 100,    # Enzyme molecules
        'S': 1000,   # Substrate molecules
        'ES': 0,     # No complex initially
        'P': 0       # No product initially
    }
    
    system = ReactionSystem(
        species=species,
        reactions=reactions,
        initial_state=initial_state,
    )
    
    return system


# ============================================================================
# L1.1: System Initialization and Validation
# ============================================================================

class TestGillespieInitialization:
    """Test proper initialization of Gillespie SSA."""
    
    def test_init_simple_system(self, simple_birth_death_system):
        """Test initialization with simple birth-death system."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        assert ssa.num_species == 1
        assert ssa.num_reactions == 2
        assert ssa.system == simple_birth_death_system
        assert ssa.species_to_index == {'S': 0}
    
    def test_init_enzymatic_system(self, enzymatic_reaction_system):
        """Test initialization with enzymatic reaction system."""
        ssa = GillespieSSA(enzymatic_reaction_system)
        
        assert ssa.num_species == 4
        assert ssa.num_reactions == 3
        assert set(ssa.species_to_index.keys()) == {'E', 'S', 'ES', 'P'}
    
    def test_stoichiometric_matrix_shape(self, simple_birth_death_system):
        """Test stoichiometric matrix has correct dimensions."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        stoich_matrix = ssa._stoichiometric_matrix
        assert stoich_matrix.shape == (2, 1)  # 2 reactions, 1 species
    
    def test_stoichiometric_matrix_values_birth_death(self, simple_birth_death_system):
        """Test stoichiometric matrix for birth-death system."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        stoich_matrix = ssa._stoichiometric_matrix
        # Birth reaction: +1 S
        assert stoich_matrix[0, 0] == 1
        # Death reaction: -1 S
        assert stoich_matrix[1, 0] == -1
    
    def test_stoichiometric_matrix_enzymatic(self, enzymatic_reaction_system):
        """Test stoichiometric matrix for enzymatic system."""
        ssa = GillespieSSA(enzymatic_reaction_system)
        
        stoich_matrix = ssa._stoichiometric_matrix
        e_idx = ssa.species_to_index['E']
        s_idx = ssa.species_to_index['S']
        es_idx = ssa.species_to_index['ES']
        p_idx = ssa.species_to_index['P']
        
        # Binding: E + S -> ES
        assert stoich_matrix[0, e_idx] == -1
        assert stoich_matrix[0, s_idx] == -1
        assert stoich_matrix[0, es_idx] == 1
        assert stoich_matrix[0, p_idx] == 0
        
        # Unbinding: ES -> E + S
        assert stoich_matrix[1, e_idx] == 1
        assert stoich_matrix[1, s_idx] == 1
        assert stoich_matrix[1, es_idx] == -1
        
        # Catalysis: ES -> E + P
        assert stoich_matrix[2, e_idx] == 1
        assert stoich_matrix[2, es_idx] == -1
        assert stoich_matrix[2, p_idx] == 1
    
    def test_validation_negative_initial_population(self):
        """Test validation catches negative initial populations."""
        species = ['S']
        reactions = [
            Reaction('birth', {}, {'S': 1}, 1.0)
        ]
        initial_state = {'S': -5}  # Invalid negative population
        
        system = ReactionSystem(species, reactions, initial_state)
        
        with pytest.raises(ValueError, match="Negative initial population"):
            GillespieSSA(system)
    
    def test_validation_undefined_species_in_reaction(self):
        """Test validation catches undefined species in reactions."""
        species = ['S']
        reactions = [
            Reaction('invalid', {'X': 1}, {'S': 1}, 1.0)  # X not in species list
        ]
        initial_state = {'S': 10}
        
        system = ReactionSystem(species, reactions, initial_state)
        
        with pytest.raises(ValueError, match="references undefined species"):
            GillespieSSA(system)

    def test_validation_rejects_incomplete_initial_state(self):
        """Every declared species must have an explicit initial population."""
        system = ReactionSystem(
            species=['A', 'B'],
            reactions=[],
            initial_state={'A': 1},
        )

        with pytest.raises(ValueError, match="initial_state keys"):
            GillespieSSA(system)

    @pytest.mark.parametrize(
        ("system", "message"),
        [
            (
                ReactionSystem(['S', 'S'], [], {'S': 1}),
                "unique non-empty names",
            ),
            (
                ReactionSystem(
                    ['S'],
                    [
                        Reaction('duplicate', {}, {'S': 1}, 1.0),
                        Reaction('duplicate', {'S': 1}, {}, 1.0),
                    ],
                    {'S': 1},
                ),
                "reaction names",
            ),
            (
                ReactionSystem(
                    ['S'],
                    [Reaction('bad_stoich', {'S': 1.5}, {}, 1.0)],
                    {'S': 2},
                ),
                "stoichiometric coefficients",
            ),
            (
                ReactionSystem(
                    ['S'],
                    [Reaction('bad_rate', {}, {'S': 1}, float('nan'))],
                    {'S': 0},
                ),
                "rate_constant",
            ),
        ],
    )
    def test_validation_rejects_malformed_reaction_system(self, system, message):
        """Reaction declarations must be finite, discrete, and unambiguous."""
        with pytest.raises(ValueError, match=message):
            GillespieSSA(system)


# ============================================================================
# L1.2: Propensity Calculations
# ============================================================================

class TestPropensityCalculations:
    """Test propensity function calculations for various reaction types."""
    
    def test_zeroth_order_propensity(self):
        """Test 0th order reaction (birth): alpha = k."""
        species = ['S']
        reactions = [
            Reaction('birth', {}, {'S': 1}, rate_constant=3.0)
        ]
        initial_state = {'S': 0}
        system = ReactionSystem(species, reactions, initial_state)
        ssa = GillespieSSA(system)
        
        populations = torch.tensor([0], dtype=torch.int64)
        propensities = ssa._calculate_propensities(populations, 0.0)
        
        # Birth rate should be constant = k
        assert torch.isclose(propensities[0], torch.tensor(3.0), atol=1e-6)
    
    def test_first_order_propensity(self):
        """Test 1st order reaction (decay): alpha = k * x."""
        species = ['S']
        reactions = [
            Reaction('decay', {'S': 1}, {}, rate_constant=2.0)
        ]
        initial_state = {'S': 10}
        system = ReactionSystem(species, reactions, initial_state)
        ssa = GillespieSSA(system)
        
        populations = torch.tensor([10], dtype=torch.int64)
        propensities = ssa._calculate_propensities(populations, 0.0)
        
        # Decay propensity = k * S = 2.0 * 10 = 20.0
        assert torch.isclose(propensities[0], torch.tensor(20.0), atol=1e-6)
    
    def test_second_order_propensity(self):
        """Test 2nd order reaction (bimolecular): alpha = k * x1 * x2."""
        species = ['A', 'B', 'C']
        reactions = [
            Reaction('combine', {'A': 1, 'B': 1}, {'C': 1}, rate_constant=0.1)
        ]
        initial_state = {'A': 20, 'B': 30, 'C': 0}
        system = ReactionSystem(species, reactions, initial_state)
        ssa = GillespieSSA(system)
        
        populations = torch.tensor([20, 30, 0], dtype=torch.int64)
        propensities = ssa._calculate_propensities(populations, 0.0)
        
        # Bimolecular propensity = k * A * B = 0.1 * 20 * 30 = 60.0
        assert torch.isclose(propensities[0], torch.tensor(60.0), atol=1e-6)
    
    def test_propensity_with_zero_population(self):
        """Test propensity is zero when reactant population is zero."""
        species = ['S']
        reactions = [
            Reaction('decay', {'S': 1}, {}, rate_constant=2.0)
        ]
        initial_state = {'S': 0}
        system = ReactionSystem(species, reactions, initial_state)
        ssa = GillespieSSA(system)
        
        populations = torch.tensor([0], dtype=torch.int64)
        propensities = ssa._calculate_propensities(populations, 0.0)
        
        # With S=0, decay propensity should be 0
        assert propensities[0] == 0.0
    
    def test_propensity_birth_death(self, simple_birth_death_system):
        """Test propensities for birth-death system."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        populations = torch.tensor([10], dtype=torch.int64)
        propensities = ssa._calculate_propensities(populations, 0.0)
        
        # Birth: rate = 5.0 (constant)
        assert torch.isclose(propensities[0], torch.tensor(5.0), atol=1e-6)
        # Death: rate = k * S = 1.0 * 10 = 10.0
        assert torch.isclose(propensities[1], torch.tensor(10.0), atol=1e-6)

    def test_fourth_order_propensity_uses_exact_combinations(self):
        """Mass-action hazards use exact binomial factors at every order."""
        ssa = GillespieSSA(
            ReactionSystem(
                species=['S'],
                reactions=[Reaction('consume_four', {'S': 4}, {}, 1.0)],
                initial_state={'S': 4},
            )
        )

        torch.testing.assert_close(
            ssa._calculate_propensities(torch.tensor([4]), 0.0),
            torch.tensor([1.0]),
        )
        torch.testing.assert_close(
            ssa._calculate_propensities(torch.tensor([3]), 0.0),
            torch.tensor([0.0]),
        )

    @pytest.mark.parametrize(
        "invalid_propensity",
        [
            torch.tensor(-1.0),
            torch.tensor(float('nan')),
            torch.tensor(float('inf')),
            torch.tensor([1.0, 2.0]),
        ],
    )
    def test_custom_propensity_must_be_finite_non_negative_scalar(
        self,
        invalid_propensity,
    ):
        """Custom hazards cannot hide invalid domain or shape semantics."""
        reaction = Reaction(
            'custom',
            {},
            {'S': 1},
            0.0,
            rate_function=lambda *_args, **_kwargs: invalid_propensity,
        )
        ssa = GillespieSSA(ReactionSystem(['S'], [reaction], {'S': 0}))

        with pytest.raises(
            ValueError,
            match="propensity must be a finite non-negative scalar",
        ):
            ssa._calculate_propensities(torch.tensor([0]), 0.0)


# ============================================================================
# L1.3: Trajectory Generation
# ============================================================================

class TestTrajectoryGeneration:
    """Test generation of stochastic trajectories."""
    
    def test_single_trajectory_birth_death(self, simple_birth_death_system):
        """Test single trajectory generation for birth-death system."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        results = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=1,
            final_time=5.0,
            seed=42
        )
        
        assert 'populations' in results
        assert 'times' in results
        assert 'reactions' in results
        assert 'trajectory_lengths' in results
        
        # Check dimensions
        assert results['populations'].shape[0] == 1  # 1 sample
        assert results['times'].shape[0] == 1
        
        # Check trajectory is valid
        traj_length = results['trajectory_lengths'][0].item()
        assert traj_length > 0
        assert results['times'][0, traj_length-1] <= 5.0  # Final time within limit

    def test_sample_uses_input_state_as_authoritative_initial_state(
        self,
        simple_birth_death_system,
    ):
        """The StochasticProcess input state is not descriptive metadata."""
        ssa = GillespieSSA(simple_birth_death_system)

        result = ssa.sample(
            input_state=torch.tensor([9]),
            num_samples=1,
            final_time=0.0,
            seed=7,
        )

        assert result['populations'][0, 0, 0].item() == 9

    def test_seeded_sample_does_not_mutate_global_torch_rng(
        self,
        simple_birth_death_system,
    ):
        """A sampler seed owns a local substream, not the process RNG."""
        ssa = GillespieSSA(simple_birth_death_system)
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()

        ssa.sample(
            input_state=torch.tensor([10]),
            num_samples=1,
            final_time=0.0,
            seed=99,
        )

        assert torch.equal(torch.random.get_rng_state(), before)

    @pytest.mark.parametrize(
        "state",
        [
            torch.tensor([-1.0]),
            torch.tensor([float('nan')]),
            torch.tensor([float('inf')]),
            torch.tensor([1.5]),
            torch.tensor([[1.0]]),
        ],
    )
    def test_sample_rejects_invalid_discrete_input_state(
        self,
        simple_birth_death_system,
        state,
    ):
        """SSA populations are a finite non-negative integer state vector."""
        ssa = GillespieSSA(simple_birth_death_system)

        with pytest.raises(ValueError, match="finite non-negative integer"):
            ssa.sample(state, num_samples=1, final_time=1.0)

    def test_impossible_custom_reaction_fails_instead_of_clamping_state(self):
        """An inconsistent custom hazard cannot create a hidden state repair."""
        reaction = Reaction(
            'consume_absent',
            {'S': 1},
            {},
            0.0,
            rate_function=lambda *_args, **_kwargs: torch.tensor(1.0e6),
        )
        ssa = GillespieSSA(ReactionSystem(['S'], [reaction], {'S': 0}))

        with pytest.raises(RuntimeError, match="negative population"):
            ssa.sample(
                torch.tensor([0]),
                num_samples=1,
                final_time=1.0,
                max_reactions=1,
                seed=5,
            )
    
    def test_multiple_trajectories(self, simple_birth_death_system):
        """Test multiple independent trajectories."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        num_samples = 10
        results = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=num_samples,
            final_time=5.0,
            seed=42
        )
        
        assert results['populations'].shape[0] == num_samples
        assert results['trajectory_lengths'].shape[0] == num_samples
        
        # All trajectories should have valid lengths
        assert torch.all(results['trajectory_lengths'] > 0)
    
    def test_trajectory_conservation(self, simple_birth_death_system):
        """Test that trajectory respects stoichiometry."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        results = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=1,
            final_time=2.0,
            seed=42
        )
        
        populations = results['populations'][0]
        reactions_fired = results['reactions'][0]
        traj_length = results['trajectory_lengths'][0].item()
        
        # Manually verify stoichiometry
        expected_pop = 10  # Initial
        for step in range(1, traj_length):
            reaction_idx = reactions_fired[step].item()
            if reaction_idx == 0:  # Birth
                expected_pop += 1
            elif reaction_idx == 1:  # Death
                expected_pop -= 1
            
            actual_pop = populations[step, 0].item()
            assert actual_pop == expected_pop, f"Population mismatch at step {step}"
    
    def test_trajectory_non_negative_populations(self, simple_birth_death_system):
        """Test that populations never go negative."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        results = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=5,
            final_time=10.0,
            seed=42
        )
        
        # Check all populations are non-negative
        assert torch.all(results['populations'] >= 0)
    
    def test_trajectory_time_monotonic(self, simple_birth_death_system):
        """Test that time is strictly increasing."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        results = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=1,
            final_time=5.0,
            seed=42
        )
        
        times = results['times'][0]
        traj_length = results['trajectory_lengths'][0].item()
        
        # Check time is monotonically increasing
        for i in range(1, traj_length):
            assert times[i] > times[i-1], "Time must be strictly increasing"
    
    def test_absorbing_state_detection(self):
        """Test that simulation stops at absorbing state."""
        # Pure death process - will reach zero and stop
        species = ['S']
        reactions = [
            Reaction('death', {'S': 1}, {}, rate_constant=1.0)
        ]
        initial_state = {'S': 5}
        system = ReactionSystem(species, reactions, initial_state)
        ssa = GillespieSSA(system)
        
        results = ssa.sample(
            input_state=torch.tensor([5.0]),
            num_samples=1,
            final_time=100.0,  # Long time
            seed=42
        )
        
        populations = results['populations'][0]
        traj_length = results['trajectory_lengths'][0].item()
        
        # Should reach zero and stop
        final_pop = populations[traj_length-1, 0].item()
        assert final_pop == 0, "Should reach absorbing state (zero population)"


# ============================================================================
# L1.4: Statistical Properties
# ============================================================================

class TestStatisticalProperties:
    """Test statistical properties of generated trajectories."""
    
    def test_birth_death_steady_state_mean(self, simple_birth_death_system):
        """Test that ensemble average approaches steady-state mean."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        # Generate many long trajectories
        num_samples = 100
        results = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=num_samples,
            final_time=20.0,  # Long time to reach steady state
            seed=42
        )
        
        # Compute final populations
        final_populations = []
        for i in range(num_samples):
            traj_length = results['trajectory_lengths'][i].item()
            final_pop = results['populations'][i, traj_length-1, 0].item()
            final_populations.append(final_pop)
        
        mean_final = torch.tensor(
            final_populations,
            dtype=torch.get_default_dtype(),
        ).mean().item()
        
        # Theoretical steady-state mean: k_birth / k_death = 5.0 / 1.0 = 5.0
        expected_mean = 5.0
        
        # Allow 20% tolerance due to finite sampling
        assert abs(mean_final - expected_mean) / expected_mean < 0.2
    
    def test_reproducibility_with_seed(self, simple_birth_death_system):
        """Test that same seed produces same trajectory."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        results1 = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=1,
            final_time=5.0,
            seed=12345
        )
        
        results2 = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=1,
            final_time=5.0,
            seed=12345
        )
        
        # Same seed should produce identical results
        assert torch.all(results1['populations'] == results2['populations'])
        assert torch.all(results1['times'] == results2['times'])
        assert torch.all(results1['reactions'] == results2['reactions'])
    
    def test_ensemble_variance(self, simple_birth_death_system):
        """Test that ensemble has expected variance."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        num_samples = 200
        results = ssa.sample(
            input_state=torch.tensor([5.0]),  # Start at expected steady state
            num_samples=num_samples,
            final_time=10.0,
            seed=42
        )
        
        # Compute variance of final populations
        final_populations = []
        for i in range(num_samples):
            traj_length = results['trajectory_lengths'][i].item()
            final_pop = results['populations'][i, traj_length-1, 0].item()
            final_populations.append(final_pop)
        
        variance = torch.tensor(
            final_populations,
            dtype=torch.get_default_dtype(),
        ).var(correction=0).item()
        
        # For birth-death at steady state, variance 鈮?mean (Poisson-like)
        expected_variance = 5.0
        
        # Allow 50% tolerance due to finite sampling and transient dynamics
        assert abs(variance - expected_variance) / expected_variance < 0.5


# ============================================================================
# L1.5: Reaction Management
# ============================================================================

class TestReactionManagement:
    """Test adding and removing reactions dynamically."""
    
    def test_add_reaction(self, simple_birth_death_system):
        """Test adding a new reaction to the system."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        initial_num_reactions = ssa.num_reactions
        assert initial_num_reactions == 2
        
        # Add a new reaction (spontaneous conversion, not useful but valid)
        new_reaction = Reaction(
            name='conversion',
            reactants={'S': 1},
            products={'S': 2},  # S doubles (unrealistic but tests mechanism)
            rate_constant=0.1
        )
        ssa.add_reaction(new_reaction)
        
        assert ssa.num_reactions == 3
        assert len(ssa.system.reactions) == 3
    
    def test_remove_reaction(self, simple_birth_death_system):
        """Test removing a reaction from the system."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        initial_num_reactions = ssa.num_reactions
        ssa.remove_reaction('birth')
        
        assert ssa.num_reactions == initial_num_reactions - 1
        assert len(ssa.system.reactions) == 1
        assert ssa.system.reactions[0].name == 'death'
    
    def test_remove_nonexistent_reaction(self, simple_birth_death_system):
        """Test that removing nonexistent reaction raises error."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        with pytest.raises(ValueError, match="Reaction not found"):
            ssa.remove_reaction('nonexistent')
    
    def test_add_reaction_with_undefined_species(self, simple_birth_death_system):
        """Test that adding reaction with undefined species raises error."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        invalid_reaction = Reaction(
            name='invalid',
            reactants={'X': 1},  # X not in species list
            products={'S': 1},
            rate_constant=1.0
        )
        
        with pytest.raises(ValueError, match="references undefined species"):
            ssa.add_reaction(invalid_reaction)


# ============================================================================
# L1.6: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_initial_population(self):
        """Test system with zero initial population (absorbing state)."""
        species = ['S']
        reactions = [
            Reaction('decay', {'S': 1}, {}, rate_constant=1.0)
        ]
        initial_state = {'S': 0}
        system = ReactionSystem(species, reactions, initial_state)
        ssa = GillespieSSA(system)
        
        results = ssa.sample(
            input_state=torch.tensor([0.0]),
            num_samples=1,
            final_time=10.0,
            seed=42
        )
        
        # Should immediately stop (no reactions possible)
        traj_length = results['trajectory_lengths'][0].item()
        assert traj_length == 1  # Only initial state

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({'num_samples': 0}, "num_samples"),
            ({'num_samples': True}, "num_samples"),
            ({'max_reactions': 0}, "max_reactions"),
            ({'max_reactions': True}, "max_reactions"),
            ({'final_time': -1.0}, "final_time"),
            ({'final_time': float('nan')}, "final_time"),
        ],
    )
    def test_sample_rejects_invalid_control_domain(
        self,
        simple_birth_death_system,
        kwargs,
        message,
    ):
        """Trajectory controls must have finite and non-degenerate domains."""
        ssa = GillespieSSA(simple_birth_death_system)

        with pytest.raises(ValueError, match=message):
            ssa.sample(torch.tensor([10]), **kwargs)

    def test_sample_rejects_seed_and_generator_together(
        self,
        simple_birth_death_system,
    ):
        """Exactly one RNG substream source may be selected."""
        ssa = GillespieSSA(simple_birth_death_system)

        with pytest.raises(ValueError, match="seed and generator"):
            ssa.sample(
                torch.tensor([10]),
                seed=3,
                generator=torch.Generator().manual_seed(4),
            )

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA generator test requires a CUDA device")
    def test_sample_accepts_index_free_cuda_generator_and_rejects_cpu_generator(
        self,
        simple_birth_death_system,
    ):
        """A generator on "cuda" serves cuda:N states; a CPU generator does not."""
        ssa = GillespieSSA(simple_birth_death_system)
        state = torch.tensor([10], device="cuda")

        result = ssa.sample(
            state,
            final_time=0.5,
            generator=torch.Generator(device="cuda").manual_seed(4),
        )
        assert result["populations"].device.type == "cuda"

        with pytest.raises(ValueError, match="generator device must match"):
            ssa.sample(state, final_time=0.5, generator=torch.Generator().manual_seed(4))

    def test_log_probability_fails_without_an_event_path(
        self,
        simple_birth_death_system,
    ):
        """Endpoint states do not identify a Markov jump path likelihood."""
        ssa = GillespieSSA(simple_birth_death_system)

        with pytest.raises(NotImplementedError, match="path likelihood"):
            ssa.log_probability(torch.tensor([10]), torch.tensor([11]))
    
    def test_very_high_reaction_rate(self):
        """Test system with very high reaction rates."""
        species = ['S']
        reactions = [
            Reaction('birth', {}, {'S': 1}, rate_constant=1000.0),
            Reaction('death', {'S': 1}, {}, rate_constant=999.0)
        ]
        initial_state = {'S': 100}
        system = ReactionSystem(species, reactions, initial_state)
        ssa = GillespieSSA(system)
        
        results = ssa.sample(
            input_state=torch.tensor([100.0]),
            num_samples=1,
            final_time=0.1,  # Short time
            max_reactions=10000,
            seed=42
        )
        
        # Should handle high rates without errors
        assert results['trajectory_lengths'][0] > 0
    
    def test_max_reactions_limit(self, simple_birth_death_system):
        """Test that max_reactions limit is respected."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        max_reactions = 50
        results = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=1,
            final_time=100.0,  # Long time
            max_reactions=max_reactions,
            seed=42
        )
        
        traj_length = results['trajectory_lengths'][0].item()
        # Should not exceed max_reactions + 1 (initial state)
        assert traj_length <= max_reactions + 1
    
    def test_get_reaction_fluxes(self, simple_birth_death_system):
        """Test calculation of average reaction fluxes."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        results = ssa.sample(
            input_state=torch.tensor([10.0]),
            num_samples=1,
            final_time=10.0,
            seed=42
        )
        
        fluxes = ssa.get_reaction_fluxes(results, sample_idx=0)
        
        assert 'birth' in fluxes
        assert 'death' in fluxes
        assert fluxes['birth'] >= 0
        assert fluxes['death'] >= 0
        # At steady state, birth flux 鈮?death flux
        # Allow large tolerance due to stochasticity
        assert abs(fluxes['birth'] - fluxes['death']) / max(fluxes['birth'], 0.1) < 2.0
    
    def test_describe_metadata(self, simple_birth_death_system):
        """Test that describe() returns valid metadata."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        metadata = ssa.describe()
        
        assert metadata.name == "GillespieSSA"
        assert "exact stochastic simulation" in metadata.description.lower()
        assert 'populations' in metadata.units
        assert 'num_species' in metadata.parameters
        assert metadata.parameters['num_species'] == 1
        assert metadata.parameters['num_reactions'] == 2
    
    def test_moments_birth_death(self, simple_birth_death_system):
        """Test moments calculation for birth-death system."""
        ssa = GillespieSSA(simple_birth_death_system)
        
        moments = ssa.moments()
        
        # Should return steady-state statistics
        assert 'steady_state_mean' in moments
        assert 'steady_state_variance' in moments
        
        # Check theoretical values
        expected_mean = 5.0  # k_birth / k_death
        assert torch.isclose(moments['steady_state_mean'], torch.tensor(expected_mean), rtol=0.01)



class TestNoFabricatedDiagnostics:
    """Fluxes and moments are defined quantities or errors (MA-21 C7).

    References: Gillespie (1977), J. Phys. Chem. 81, 2340-2361; Gardiner (2009),
    Stochastic Methods, Sect. 11.1 (immigration-death process).
    Mathematical expression: mean flux J_k = N_k / T over the observation
    horizon T; stationary immigration-death counts ~ Poisson(k_b / k_d).
    """

    def test_flux_is_averaged_over_the_horizon_not_the_last_event(self, simple_birth_death_system):
        ssa = GillespieSSA(simple_birth_death_system)
        results = ssa.sample(torch.tensor([10.0]), num_samples=1, final_time=10.0, seed=42)
        length = int(results["trajectory_lengths"][0])
        births = int((results["reactions"][0, 1:length] == 0).sum())

        fluxes = ssa.get_reaction_fluxes(results, sample_idx=0)

        assert results["final_time"] == pytest.approx(10.0)
        assert fluxes["birth"] == pytest.approx(births / 10.0)

    def test_flux_of_a_truncated_trajectory_is_an_error(self, simple_birth_death_system):
        ssa = GillespieSSA(simple_birth_death_system)
        results = ssa.sample(
            torch.tensor([10.0]), num_samples=1, final_time=100.0, max_reactions=5, seed=42
        )

        assert bool(results["reached_max_reactions"][0])
        with pytest.raises(ValueError, match="max_reactions"):
            ssa.get_reaction_fluxes(results, sample_idx=0)

    def test_unsupported_system_moments_raise_instead_of_an_empty_mapping(self):
        system = ReactionSystem(
            ["A", "B"],
            [
                Reaction("dimerize", {"A": 2}, {"B": 1}, 0.1),
                Reaction("split", {"B": 1}, {"A": 2}, 0.2),
            ],
            {"A": 10, "B": 0},
        )
        with pytest.raises(NotImplementedError, match="moments"):
            GillespieSSA(system).moments()

    def test_two_reaction_system_that_is_not_immigration_death_is_not_given_poisson_moments(self):
        """A second-order death is not the linear immigration-death process."""
        system = ReactionSystem(
            ["S"],
            [
                Reaction("birth", {}, {"S": 1}, 5.0),
                Reaction("pair_death", {"S": 2}, {}, 1.0),
            ],
            {"S": 10},
        )
        with pytest.raises(NotImplementedError, match="moments"):
            GillespieSSA(system).moments()


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])



class TestSeedRecording:
    """An unseeded run records the seed it drew, so the run can be reproduced."""

    def test_drawn_seed_is_recorded_and_reproduces_the_run(self, simple_birth_death_system):
        ssa = GillespieSSA(simple_birth_death_system)
        first = ssa.sample(torch.tensor([10.0]), num_samples=2, final_time=5.0)

        assert isinstance(first["seed"], int)
        again = ssa.sample(torch.tensor([10.0]), num_samples=2, final_time=5.0, seed=first["seed"])
        assert torch.equal(first["populations"], again["populations"])

    def test_explicit_seed_is_recorded(self, simple_birth_death_system):
        result = GillespieSSA(simple_birth_death_system).sample(
            torch.tensor([10.0]), num_samples=1, final_time=1.0, seed=17
        )
        assert result["seed"] == 17

    def test_caller_generator_has_no_recorded_seed(self, simple_birth_death_system):
        result = GillespieSSA(simple_birth_death_system).sample(
            torch.tensor([10.0]), num_samples=1, final_time=1.0,
            generator=torch.Generator().manual_seed(3),
        )
        assert result["seed"] is None


class TestRound5CountBasedPropensities:
    """ReactionSystem carries no volume: the SSA reads count-based c_mu only."""

    @staticmethod
    def _birth():
        return [Reaction(name="birth", reactants={}, products={"S": 1}, rate_constant=1.0)]

    def test_reaction_system_has_no_volume_field(self):
        with pytest.raises(TypeError, match="volume"):
            ReactionSystem(["S"], self._birth(), {"S": 0}, volume=1.0)
        assert not hasattr(ReactionSystem(["S"], self._birth(), {"S": 0}), "volume")

    def test_metadata_states_the_count_based_propensity_convention(self):
        ssa = GillespieSSA(ReactionSystem(["S"], self._birth(), {"S": 0}))
        parameters = ssa.describe().parameters
        assert "volume" not in parameters
        assert parameters["propensity_convention"] == "count_based_stochastic_rate_constants"
