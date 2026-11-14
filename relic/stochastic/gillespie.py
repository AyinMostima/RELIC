"""Gillespie direct-method sampler for declared discrete reaction systems.

This module is the numerical algorithm owner for a continuous-time Markov
jump process. It does not define a biogeochemical process law or provide a
production-runtime bypass around GraphIR, FunctorBase, or the Simulator.

Mathematical framework:
    Count state: x in non-negative integer species counts.
    Reaction j: reactant vector nu_minus_j and net change nu_j.
    Hazard: a_j(x) = c_j prod_i binom(x_i, nu_minus_ij).
    Total hazard: a_0(x) = sum_j a_j(x).
    Waiting time: tau = -log(U_1) / a_0(x).
    Selection: j = min{k: sum_{l <= k} a_l(x) > U_2 a_0(x)}.

References:
    - Gillespie, D.T. (1976). A general method for numerically simulating
      the stochastic time evolution of coupled chemical reactions.
      J. Comput. Phys. 22, 403-434.
    - Gillespie, D.T. (1977). Exact stochastic simulation of coupled
      chemical reactions. J. Phys. Chem. 81, 2340-2361.
    - Gibson, M.A. & Bruck, J. (2000). Efficient exact stochastic simulation
      of chemical systems. J. Phys. Chem. A 104, 1876-1889.

Authors: Yuchong Long
"""

import logging
import secrets
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .category import StochasticProcess, ProcessMetadata

logger = logging.getLogger(__name__)


@dataclass
class Reaction:
    """
    Definition of a single biochemical reaction.
    
    Parameters:
        name (str): Reaction identifier (e.g., 'photosynthesis', 'respiration')
        reactants (Dict[str, int]): Reactant species and stoichiometric coefficients
        products (Dict[str, int]): Product species and stoichiometric coefficients  
        rate_constant (float): Reaction rate constant [appropriate units]
        rate_function (Callable, optional): Custom propensity function alpha(x)
        
    Mathematical Expression:
        Generic reaction: a_1S_1 + a_2S_2 -> b_1P_1 + b_2P_2
        Count-based mass action:
        alpha(x) = c * product_i binom(x_i, a_i)
        
    Examples:
        Michaelis-Menten: E + S -> ES, alpha(x) = k_1*E*S
        Decay: S -> empty_set, alpha(x) = k*S  
        Birth: empty_set -> S, alpha(x) = k (constant)
        
    Units:
        rate_constant units depend on reaction order:
        - 0th order: [species/time] (e.g., molecules/s)
        - 1st order: [1/time] (e.g., s-1)
        - 2nd order: [1/(species*time)] (e.g., molecules-1*s-1)
    """
    name: str
    reactants: Dict[str, int]  # {species_name: stoichiometry}
    products: Dict[str, int]
    rate_constant: float
    rate_function: Optional[Callable] = None


@dataclass
class ReactionSystem:
    """
    Complete biochemical reaction system definition.
    
    Parameters:
        species (List[str]): Names of all chemical species
        reactions (List[Reaction]): All reactions in the system
        initial_state (Dict[str, int]): Initial populations for each species

    Propensity convention (count-based):
        Every rate_constant is the stochastic reaction parameter c_mu of
        Gillespie (1976, 1977) in count units, and the propensity is
        a_mu(x) = c_mu h_mu(x) with h_mu(x) = product_i binom(x_i, a_i), the
        number of distinct reactant combinations. No system size (volume)
        enters the SSA: a deterministic concentration-based rate constant is
        converted to c_mu by the caller (Gillespie 1977) before it is
        declared, so the system carries no volume field.

    Units:
        Species populations: number of molecules (counts)
        rate_constant: counts^(1 - m) per time for a reaction of order m
        Time: the time unit of the rate constants (e.g. s)
    """
    species: List[str]
    reactions: List[Reaction]
    initial_state: Dict[str, int]


class GillespieSSA(StochasticProcess):
    """
    Gillespie Stochastic Simulation Algorithm implementation.
    
    This class provides exact stochastic simulation of coupled biochemical
    reactions using the direct method (Gillespie 1976). The algorithm 
    generates statistically exact sample paths of the underlying Markov 
    jump process without approximation.
    
    Key Features:
        - Exact (non-approximate) stochastic simulation
        - Support for arbitrary reaction networks
        - Mass action and custom propensity functions
        - Efficient direct method implementation
        - Local PyTorch random-number substreams
        
    Algorithm Steps:
        1. Calculate all propensities alpha_j(x) and their sum lambda(x)
        2. Generate tau ~ Exp(lambda(x)) for time to next reaction
        3. Select reaction k with probability alpha_k(x)/lambda(x)
        4. Update state: x -> x + nu_k, time: t -> t + tau
        5. Repeat until simulation end time
        
    Mathematical Foundation:
        The algorithm is based on the exact solution of the Chemical Master
        Equation through Monte Carlo sampling of the underlying continuous-time
        Markov chain. Each trajectory is a statistically exact realization.
    """
    
    def __init__(self, reaction_system: ReactionSystem):
        """
        Initialize Gillespie SSA with reaction system.
        
        Parameters:
            reaction_system (ReactionSystem): Complete reaction system definition
        """
        self._validate_system(reaction_system)
        self.system = reaction_system
        self.num_species = len(self.system.species)
        self.num_reactions = len(self.system.reactions)

        self.species_to_index = {
            species: idx for idx, species in enumerate(self.system.species)
        }

        self._stoichiometric_matrix = self._build_stoichiometric_matrix()

        logger.info(
            "Initialized Gillespie SSA: %s species, %s reactions",
            self.num_species,
            self.num_reactions,
        )

    def _build_stoichiometric_matrix(self) -> Tensor:
        """
        Build the net stoichiometric matrix for the reaction system.
        
        Returns:
            Tensor: Matrix with shape [num_reactions, num_species], where
                nu[k, i] is the net count change caused by reaction k.
        """
        matrix = torch.zeros(self.num_reactions, self.num_species, dtype=torch.int64)
        
        for k, reaction in enumerate(self.system.reactions):
            for species, coeff in reaction.reactants.items():
                i = self.species_to_index[species]
                matrix[k, i] -= coeff

            for species, coeff in reaction.products.items():
                i = self.species_to_index[species]
                matrix[k, i] += coeff

        return matrix

    @staticmethod
    def _as_finite_scalar(value: object, name: str) -> Tensor:
        """Return a finite scalar tensor without accepting Boolean values."""
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a finite scalar")
        try:
            scalar = torch.as_tensor(value, dtype=torch.get_default_dtype())
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{name} must be a finite scalar") from exc
        if scalar.ndim != 0 or not bool(torch.isfinite(scalar).item()):
            raise ValueError(f"{name} must be a finite scalar")
        return scalar

    @classmethod
    def _validate_count(cls, value: object, name: str) -> None:
        """Validate one declared non-negative integer population."""
        scalar = cls._as_finite_scalar(value, name)
        if bool((scalar < 0).item()):
            raise ValueError(f"Negative initial population for {name}: {value}")
        if not bool(torch.equal(scalar, torch.round(scalar))):
            raise ValueError(f"Initial population for {name} must be an integer")

    @staticmethod
    def _validate_stoichiometry(mapping: object, reaction_name: str) -> None:
        """Validate one reactant or product count mapping."""
        if not isinstance(mapping, dict):
            raise ValueError(
                f"Reaction {reaction_name} stoichiometric coefficients "
                "must be positive integers"
            )
        for coefficient in mapping.values():
            if (
                isinstance(coefficient, bool)
                or not isinstance(coefficient, int)
                or coefficient <= 0
            ):
                raise ValueError(
                    f"Reaction {reaction_name} stoichiometric coefficients "
                    "must be positive integers"
                )

    @classmethod
    def _validate_reaction(
        cls,
        reaction: object,
        species_names: set[str],
    ) -> None:
        """Validate one declared reaction without changing the system."""
        if not isinstance(reaction, Reaction):
            raise ValueError("All reactions must be Reaction declarations")
        if not isinstance(reaction.name, str) or not reaction.name.strip():
            raise ValueError("reaction names must be unique non-empty strings")

        cls._validate_stoichiometry(reaction.reactants, reaction.name)
        cls._validate_stoichiometry(reaction.products, reaction.name)

        referenced = set(reaction.reactants) | set(reaction.products)
        missing = referenced - species_names
        if missing:
            missing_text = ", ".join(sorted(str(name) for name in missing))
            raise ValueError(
                f"Reaction {reaction.name} references undefined species: {missing_text}"
            )

        rate_constant = cls._as_finite_scalar(
            reaction.rate_constant,
            f"Reaction {reaction.name} rate_constant",
        )
        if bool((rate_constant < 0).item()):
            raise ValueError(
                f"Reaction {reaction.name} rate_constant must be non-negative"
            )
        if reaction.rate_function is not None and not callable(reaction.rate_function):
            raise ValueError(f"Reaction {reaction.name} rate_function must be callable")

    @classmethod
    def _validate_system(cls, system: object) -> None:
        """Validate the complete declaration without implicit repair."""
        if not isinstance(system, ReactionSystem):
            raise TypeError("reaction_system must be a ReactionSystem")
        if (
            not isinstance(system.species, list)
            or not system.species
            or any(not isinstance(name, str) or not name.strip() for name in system.species)
            or len(set(system.species)) != len(system.species)
        ):
            raise ValueError("Species must have unique non-empty names")
        if not isinstance(system.reactions, list):
            raise ValueError("reactions must be a list of Reaction declarations")
        if not isinstance(system.initial_state, dict):
            raise ValueError("initial_state keys must exactly match declared species")

        species_names = set(system.species)
        if set(system.initial_state) != species_names:
            raise ValueError("initial_state keys must exactly match declared species")
        for species, population in system.initial_state.items():
            cls._validate_count(population, species)

        reaction_names: set[str] = set()
        for reaction in system.reactions:
            cls._validate_reaction(reaction, species_names)
            if reaction.name in reaction_names:
                raise ValueError("reaction names must be unique non-empty strings")
            reaction_names.add(reaction.name)

    def sample(
        self,
        input_state: Tensor,
        num_samples: int = 1,
        seed: Optional[int] = None,
        final_time: float = 10.0,
        max_reactions: int = 100000,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate exact stochastic trajectories using Gillespie SSA.
        
        Parameters:
            input_state (Tensor): Authoritative initial species count vector
            num_samples (int): Number of independent trajectory samples
            seed (int, optional): Seed for a private PyTorch RNG substream
            final_time (float): Simulation end time [time units of rate constants]
            max_reactions (int): Maximum number of reactions per trajectory
            generator (torch.Generator, optional): Caller-owned RNG substream
            **kwargs: Additional parameters (e.g., environmental conditions)
            
        Returns:
            Dict[str, Tensor]: Simulation results
                'populations': Species populations [shape: (num_samples, max_steps, num_species)]
                'times': Reaction times [shape: (num_samples, max_steps)]
                'reactions': Reaction indices fired [shape: (num_samples, max_steps)]
                'propensities': Propensity values [shape: (num_samples, max_steps, num_reactions)]
                'trajectory_lengths': Actual trajectory lengths [shape: (num_samples,)]
                
        Mathematical Implementation:
            For each trajectory:
            1. tau = -log(U_1)/a_0(x), with U_1 uniform on (0, 1)
            2. k = min{m: sum_{j <= m} a_j(x) > U_2 a_0(x)}
            3. x <- x + nu_k and t <- t + tau
        """
        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if (
            isinstance(max_reactions, bool)
            or not isinstance(max_reactions, int)
            or max_reactions <= 0
        ):
            raise ValueError("max_reactions must be a positive integer")

        final_time_tensor = self._as_finite_scalar(final_time, "final_time")
        if bool((final_time_tensor < 0).item()):
            raise ValueError("final_time must be a finite non-negative scalar")
        final_time_value = float(final_time_tensor.item())

        if not isinstance(input_state, Tensor):
            raise ValueError("input_state must be a finite non-negative integer vector")
        if (
            input_state.ndim != 1
            or input_state.shape[0] != self.num_species
            or input_state.dtype == torch.bool
            or input_state.is_complex()
            or not bool(torch.all(torch.isfinite(input_state)).item())
            or bool(torch.any(input_state < 0).item())
            or (
                torch.is_floating_point(input_state)
                and not bool(torch.equal(input_state, torch.round(input_state)))
            )
        ):
            raise ValueError("input_state must be a finite non-negative integer vector")

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator")
        if seed is not None and generator is not None:
            raise ValueError("seed and generator are mutually exclusive")

        device = input_state.device
        # The seed that initializes the local stream is recorded in the result
        # (an explicit seed, or the one drawn when none is given) so every run
        # can be reproduced. A caller-supplied generator carries the caller's
        # state; no seed is recorded for it.
        used_seed: Optional[int] = None
        if generator is None:
            used_seed = seed if seed is not None else secrets.randbits(63)
            local_generator = torch.Generator(device=device)
            local_generator.manual_seed(used_seed)
        else:
            generator_device = torch.device(generator.device)
            # A generator created on "cuda" (no index) serves states on the
            # current CUDA device; only an explicit index mismatch is an error.
            if generator_device.type != device.type or (
                generator_device.index is not None
                and device.index is not None
                and generator_device.index != device.index
            ):
                raise ValueError("generator device must match input_state device")
            local_generator = generator

        float_dtype = torch.get_default_dtype()
        max_steps = max_reactions + 1
        populations = torch.zeros(
            num_samples,
            max_steps,
            self.num_species,
            dtype=torch.int64,
            device=device,
        )
        times = torch.zeros(
            num_samples,
            max_steps,
            dtype=float_dtype,
            device=device,
        )
        reactions = torch.full(
            (num_samples, max_steps),
            -1,
            dtype=torch.int64,
            device=device,
        )
        propensities = torch.zeros(
            num_samples,
            max_steps,
            self.num_reactions,
            dtype=float_dtype,
            device=device,
        )
        trajectory_lengths = torch.zeros(
            num_samples,
            dtype=torch.int64,
            device=device,
        )
        # True where a trajectory stopped at max_reactions before final_time:
        # it then covers only part of the observation horizon.
        reached_max_reactions = torch.zeros(num_samples, dtype=torch.bool, device=device)

        initial_pop = input_state.to(device=device, dtype=torch.int64)
        stoichiometric_matrix = self._stoichiometric_matrix.to(device=device)

        # References: Gillespie (1976, 1977), direct SSA method.
        # Formula: tau=-log(U1)/a0; select min k with cumsum(a)[k]>U2*a0.
        # Parameters: a is the hazard vector, a0 its sum, and nu is net stoichiometry.
        for sample_idx in range(num_samples):
            current_pop = initial_pop.clone()
            current_time = 0.0
            step = 0

            populations[sample_idx, 0] = current_pop
            times[sample_idx, 0] = current_time

            while current_time < final_time_value and step < max_reactions:
                props = self._calculate_propensities(current_pop, current_time, **kwargs)
                propensities[sample_idx, step] = props

                total_hazard = torch.sum(props)
                if bool((total_hazard == 0).item()):
                    logger.debug(
                        "Sample %s: no reactions possible at step %s",
                        sample_idx,
                        step,
                    )
                    break

                uniform_time = torch.rand(
                    (),
                    dtype=float_dtype,
                    device=device,
                    generator=local_generator,
                ).clamp_min(torch.finfo(float_dtype).tiny)
                uniform_reaction = torch.rand(
                    (),
                    dtype=float_dtype,
                    device=device,
                    generator=local_generator,
                )
                tau = -torch.log(uniform_time) / total_hazard
                next_time = current_time + float(tau.item())

                if next_time >= final_time_value:
                    break

                threshold = uniform_reaction * total_hazard
                cumulative_hazard = torch.cumsum(props, dim=0)
                reaction_idx = int(
                    torch.searchsorted(
                        cumulative_hazard,
                        threshold,
                        right=True,
                    ).item()
                )
                if reaction_idx >= self.num_reactions:
                    raise RuntimeError("SSA reaction selection exceeded the hazard domain")

                stoich_vector = stoichiometric_matrix[reaction_idx]
                new_pop = current_pop + stoich_vector

                if bool(torch.any(new_pop < 0).item()):
                    raise RuntimeError(
                        "SSA reaction produced a negative population; "
                        "the declared propensity is inconsistent with stoichiometry"
                    )

                step += 1
                current_pop = new_pop
                current_time = next_time

                populations[sample_idx, step] = current_pop
                times[sample_idx, step] = current_time
                reactions[sample_idx, step] = reaction_idx

            trajectory_lengths[sample_idx] = step + 1
            reached_max_reactions[sample_idx] = (
                step >= max_reactions and current_time < final_time_value
            )

        return {
            "populations": populations,
            "times": times,
            "reactions": reactions,
            "propensities": propensities,
            "trajectory_lengths": trajectory_lengths,
            "reached_max_reactions": reached_max_reactions,
            "final_time": final_time_value,
            "seed": used_seed,
            "species_names": self.system.species,
            "reaction_names": [reaction.name for reaction in self.system.reactions],
            "stoichiometric_matrix": stoichiometric_matrix,
        }

    def _calculate_propensities(self, populations: Tensor, time: float, **kwargs) -> Tensor:
        """
        Calculate propensity functions for all reactions.
        
        Parameters:
            populations (Tensor): Current species populations [num_species]
            time (float): Current simulation time
            **kwargs: Environmental conditions (temperature, pH, etc.)
            
        Returns:
            Tensor: Propensity values [num_reactions]
            
        Mathematical expression:
            Count-based mass action hazard:
            a_j(x) = c_j product_i binom(x_i, nu_minus_ij).
        """
        if (
            not isinstance(populations, Tensor)
            or populations.ndim != 1
            or populations.shape[0] != self.num_species
            or populations.dtype == torch.bool
            or populations.is_complex()
            or not bool(torch.all(torch.isfinite(populations)).item())
            or bool(torch.any(populations < 0).item())
            or (
                torch.is_floating_point(populations)
                and not bool(torch.equal(populations, torch.round(populations)))
            )
        ):
            raise ValueError("populations must be a finite non-negative integer vector")

        dtype = torch.get_default_dtype()
        device = populations.device
        propensities = torch.zeros(self.num_reactions, dtype=dtype, device=device)

        # References: Gillespie (1976, 1977), stochastic mass-action hazard.
        # Formula: a_j(x)=c_j*product_i binom(x_i, nu_minus_ij).
        # Parameters: c_j is the microscopic hazard constant and x_i is a count.
        for k, reaction in enumerate(self.system.reactions):
            if reaction.rate_function is not None:
                try:
                    propensity = torch.as_tensor(
                        reaction.rate_function(populations, time, **kwargs),
                        dtype=dtype,
                        device=device,
                    )
                except (TypeError, ValueError, RuntimeError) as exc:
                    raise ValueError(
                        f"Reaction {reaction.name} propensity must be a finite "
                        "non-negative scalar"
                    ) from exc
            else:
                propensity = torch.as_tensor(
                    reaction.rate_constant,
                    dtype=dtype,
                    device=device,
                )

                for species, coeff in reaction.reactants.items():
                    species_idx = self.species_to_index[species]
                    population = populations[species_idx].to(dtype=dtype)
                    if bool((population < coeff).item()):
                        propensity = torch.zeros((), dtype=dtype, device=device)
                        break
                    falling_terms = population - torch.arange(
                        coeff,
                        dtype=dtype,
                        device=device,
                    )
                    factorial_terms = torch.arange(
                        1,
                        coeff + 1,
                        dtype=dtype,
                        device=device,
                    )
                    propensity = propensity * (
                        torch.prod(falling_terms) / torch.prod(factorial_terms)
                    )

            if (
                propensity.ndim != 0
                or not bool(torch.isfinite(propensity).item())
                or bool((propensity < 0).item())
            ):
                raise ValueError(
                    f"Reaction {reaction.name} propensity must be a finite "
                    "non-negative scalar"
                )
            propensities[k] = propensity

        return propensities

    def log_probability(
        self,
        input_state: Tensor,
        output_state: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        Reject endpoint-only likelihood requests.

        A Markov jump path likelihood requires the event times, reaction
        identities, and observation horizon. Endpoint states alone are not a
        sufficient statistic for that path probability.
        """
        del input_state, output_state, kwargs
        raise NotImplementedError(
            "SSA path likelihood requires event times, reaction identities, "
            "and the observation horizon"
        )

    def moments(self, input_state: Optional[Tensor] = None, **kwargs) -> Dict[str, Tensor]:
        """
        Theoretical moments of the linear immigration-death process.

        Only the one-species system  empty -> S (rate k_b) and S -> empty
        (first order, rate k_d), both mass action, has closed-form moments
        here. Any other network raises NotImplementedError: its moments must
        be estimated from sample paths or a moment closure, never reported
        as an empty mapping.
        """
        del input_state, kwargs
        birth, death = self._immigration_death_reactions()

        # References: Gardiner (2009), Stochastic Methods, Sect. 11.1.
        # Mathematical expression: stationary X ~ Poisson(k_b / k_d), so
        #     E[X] = Var[X] = k_b / k_d.
        # Parameter meanings: k_b = immigration (zeroth-order) rate constant,
        #     k_d = first-order death rate constant.
        # Units: k_b in molecules per time, k_d per time, E[X] in molecules and
        #     Var[X] in molecules^2.
        # Domain/constraints: k_d > 0 (otherwise no stationary distribution).
        # Sign convention: births add one molecule, deaths remove one.
        birth_rate = torch.as_tensor(birth.rate_constant, dtype=torch.get_default_dtype())
        death_rate = torch.as_tensor(death.rate_constant, dtype=torch.get_default_dtype())
        if not bool((death_rate > 0).item()):
            raise ValueError(
                "The immigration-death process has no stationary distribution "
                f"without a positive death rate, got {float(death_rate)}"
            )
        steady_mean = birth_rate / death_rate
        return {
            "steady_state_mean": steady_mean,
            "steady_state_variance": steady_mean.clone(),
            "birth_rate": birth_rate,
            "death_rate": death_rate,
        }

    def _immigration_death_reactions(self) -> Tuple[Reaction, Reaction]:
        """Return (birth, death) when the system is a linear immigration-death process."""
        reactions = self.system.reactions
        if len(self.system.species) == 1 and len(reactions) == 2:
            species = self.system.species[0]
            births = [
                r for r in reactions
                if not r.reactants and r.products == {species: 1} and r.rate_function is None
            ]
            deaths = [
                r for r in reactions
                if r.reactants == {species: 1} and not r.products and r.rate_function is None
            ]
            if len(births) == 1 and len(deaths) == 1:
                return births[0], deaths[0]
        raise NotImplementedError(
            "Closed-form moments are implemented only for the linear "
            "immigration-death process (empty -> S, S -> empty, mass action); "
            "estimate moments of other networks from sample paths"
        )

    def describe(self) -> ProcessMetadata:
        """Return Gillespie SSA metadata."""
        return ProcessMetadata(
            name="GillespieSSA",
            description=(
                "Exact stochastic simulation by the numerical direct method "
                "for a declared discrete reaction network"
            ),
            units={
                'populations': 'molecules',
                'times': 'time_units_of_rate_constants',
                'propensities': 'reactions_per_time'
            },
            parameters={
                'num_species': self.num_species,
                'num_reactions': self.num_reactions,
                "propensity_convention": "count_based_stochastic_rate_constants",
                "species": self.system.species,
            },
            references="Gillespie (1976, 1977), Gibson & Bruck (2000)"
        )

    def add_reaction(self, reaction: Reaction):
        """
        Add a new reaction to the system.
        
        Parameters:
            reaction (Reaction): New reaction to add
        """
        self._validate_reaction(reaction, set(self.system.species))
        if any(existing.name == reaction.name for existing in self.system.reactions):
            raise ValueError("reaction names must be unique non-empty strings")

        self.system.reactions.append(reaction)
        self.num_reactions = len(self.system.reactions)
        self._stoichiometric_matrix = self._build_stoichiometric_matrix()

        logger.info("Added reaction: %s", reaction.name)

    def remove_reaction(self, reaction_name: str):
        """
        Remove a reaction from the system.
        
        Parameters:
            reaction_name (str): Name of reaction to remove
        """
        if not isinstance(reaction_name, str) or not reaction_name.strip():
            raise ValueError("reaction_name must be a non-empty string")

        original_count = len(self.system.reactions)
        self.system.reactions = [r for r in self.system.reactions if r.name != reaction_name]

        if len(self.system.reactions) == original_count:
            raise ValueError(f"Reaction not found: {reaction_name}")

        self.num_reactions = len(self.system.reactions)
        self._stoichiometric_matrix = self._build_stoichiometric_matrix()

        logger.info("Removed reaction: %s", reaction_name)

    def get_reaction_fluxes(self, trajectory_data: Dict[str, Tensor], 
                           sample_idx: int = 0) -> Dict[str, float]:
        """
        Calculate average reaction fluxes from trajectory data.
        
        Parameters:
            trajectory_data (Dict[str, Tensor]): Output from sample() method
            sample_idx (int): Which sample trajectory to analyze
            
        Returns:
            Dict[str, float]: Average reaction flux for each reaction [reactions/time]
        """
        reactions_fired = trajectory_data['reactions'][sample_idx]
        traj_length = int(trajectory_data['trajectory_lengths'][sample_idx].item())
        horizon = float(trajectory_data['final_time'])
        if bool(trajectory_data['reached_max_reactions'][sample_idx]):
            raise ValueError(
                f"Trajectory {sample_idx} stopped at max_reactions before final_time; "
                "its reaction counts do not cover the observation horizon"
            )
        if not horizon > 0.0:
            raise ValueError(f"Mean reaction fluxes need a positive horizon, got {horizon}")

        # References: Gillespie (1977), J. Phys. Chem. 81, 2340-2361.
        # Mathematical expression: J_k = N_k(T) / T.
        # Parameter meanings: N_k(T) = number of firings of reaction k in
        #     [0, T], T = final_time of the sample call (observation horizon).
        # Units: J_k in reactions per time unit of the rate constants.
        # Domain/constraints: T > 0 and a trajectory that covers [0, T] (not
        #     stopped by max_reactions); zero firings give J_k = 0 legitimately.
        #     The time of the last event is not the horizon.
        # Sign convention: J_k >= 0 counts firings of reaction k.
        reaction_counts = torch.zeros(self.num_reactions)
        for step in range(1, traj_length):  # Skip initial state
            reaction_idx = int(reactions_fired[step].item())
            if reaction_idx >= 0:
                reaction_counts[reaction_idx] += 1

        return {
            reaction.name: reaction_counts[k].item() / horizon
            for k, reaction in enumerate(self.system.reactions)
        }
