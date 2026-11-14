"""
Stochastic Processes and Uncertainty Framework for RELIC

This module provides a comprehensive framework for handling stochasticity and 
uncertainty in biogeochemical simulations. It implements unified interfaces
for random processes, parameter uncertainty quantification, and validation
metrics following CF-1.8 standards.

Core Components:
    category: Abstract base classes for stochastic processes
    distributions: Parameter distributions and spatial correlation
    ensemble_runner: Batch execution and statistical aggregation
    validation: PIT, CRPS, and coverage probability metrics
    weather_generator: Stochastic weather generation (OU + Markov-Gamma)
    sde_solver: Stochastic differential equation solver (Euler-Maruyama)
    gillespie: Gillespie Stochastic Simulation Algorithm (SSA)

References:
    - Gneiting, T. and Raftery, A.E. (2007). Strictly proper scoring rules,
      prediction, and estimation. J. Am. Stat. Assoc. 102, 359-378.
    - Gillespie, D.T. (1976). A general method for numerically simulating
      the stochastic time evolution of coupled chemical reactions.
      J. Comput. Phys. 22, 403-434.
    - Hasselmann, K. (1976). Stochastic climate models. Part I. Theory.
      Tellus 28, 473-485.
    - Kloeden, P.E. and Platen, E. (1992). Numerical Solution of
      Stochastic Differential Equations. Springer.
    - Wilks, D.S. (2006). Statistical Methods in the Atmospheric Sciences.
      Academic Press, 2nd edition.

Authors: Yuchong Long
License: MIT
Version: 1.0.0
"""
from typing import Any, Dict

from .category import StochasticProcess, ValidationMetrics
from .distributions import ParameterDistribution, SpatialCorrelation, sample_with_correlation
from .ensemble_runner import EnsembleRunner, EnsembleResults
from .validation import (
    brier_score,
    coverage_probability,
    crps_score,
    pit_test,
    quantile_score,
)
from .weather_generator import (
    MarkovGammaPrecipitation,
    OrnsteinUhlenbeckTemperature,
    WeatherGenerator,
)
from .sde_solver import EulerMaruyamaSolver
from .gillespie import GillespieSSA

def get_available_distributions() -> Dict[str, Dict[str, Any]]:
    """Return available distribution components for health checks."""
    return {
        "ParameterDistribution": {
            "available": True,
            "error": None,
        },
        "SpatialCorrelation": {
            "available": True,
            "error": None,
        },
    }


def _bound_process_names(semantic: str) -> list[str]:
    """Return the bound process names of one stochastic semantic.

    The import is deferred because relic.stochastic.workflows binds scientific
    Functors and the orchestration owners, which the package initializer must
    not import.
    """
    from relic.stochastic.workflows import BOUND_STOCHASTIC_PROCESSES

    return sorted(
        name
        for name, binding in BOUND_STOCHASTIC_PROCESSES.items()
        if binding.semantic == semantic
    )


def get_available_samplers() -> Dict[str, Dict[str, Any]]:
    """Return available sampling/execution components for health checks."""
    from relic.stochastic.workflows import (
        BOUND_CONTINUOUS_SDE_GRAPH_OWNER,
        BOUND_MARKED_JUMP_GRAPH_OWNER,
    )

    return {
        "EnsembleRunner": {
            "available": True,
            "error": None,
        },
        "WeatherGenerator": {
            "available": True,
            "error": None,
            "contract": "numeric_only",
            "runtime_ready": False,
            "status": "[N]",
            "requirements": [
                "declared weather-forcing operator",
                "validated GraphIR/Functor runtime binding",
                "StateManager forcing/state binding",
                "ParameterManager and unit binding",
                "simulator calendar and forcing-cursor binding",
                "durable simulator checkpoint binding",
                "restartable stochastic substream binding",
            ],
        },
        "EulerMaruyamaSolver": {
            "available": True,
            "error": None,
            # The standalone class stays numeric-only; runtime readiness is
            # carried by the bound scientific Functor/GraphIR processes.
            "contract": "numeric_only",
            "runtime_ready": True,
            "status": "bound",
            "bound_processes": _bound_process_names("continuous_sde"),
            "graph_owner": BOUND_CONTINUOUS_SDE_GRAPH_OWNER,
            "requirements": [
                "declared drift operator",
                "declared diffusion operator",
                "validated GraphIR/Functor runtime binding",
                "restartable stochastic substream binding",
            ],
        },
        "GillespieSSA": {
            "available": True,
            "error": None,
            # The standalone SSA stays numeric-only; runtime readiness is
            # carried by the bound marked-jump Functor/GraphIR processes.
            # Arbitrary reaction-network declarations remain unbound.
            "contract": "numeric_only",
            "runtime_ready": True,
            "status": "bound",
            "bound_processes": _bound_process_names("marked_jump"),
            "graph_owner": BOUND_MARKED_JUMP_GRAPH_OWNER,
            "requirements": [
                "declared jump/reaction operator",
                "declared propensity and stoichiometric reset map",
                "conservation matrix or declared open-system route",
                "validated GraphIR/Functor runtime binding",
                "StateManager initial-state binding",
                "ParameterManager and unit binding",
                "restartable stochastic substream binding",
            ],
        },
    }

__all__ = [
    # Abstract interfaces
    'StochasticProcess',
    'ValidationMetrics',
    
    # Parameter distributions
    'ParameterDistribution',
    'SpatialCorrelation',
    'sample_with_correlation',
    
    # Ensemble execution
    'EnsembleRunner', 
    'EnsembleResults',
    
    # Validation metrics
    'pit_test',
    'crps_score',
    'coverage_probability', 
    'brier_score',
    'quantile_score',
    
    # Weather generation
    'OrnsteinUhlenbeckTemperature',
    'MarkovGammaPrecipitation',
    'WeatherGenerator',
    
    # Stochastic processes
    'EulerMaruyamaSolver',
    'GillespieSSA',
    'get_available_distributions',
    'get_available_samplers',
]

__version__ = '1.0.0'
__author__ = 'Yuchong Long'
