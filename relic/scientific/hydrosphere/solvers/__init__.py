"""Numerical solver utilities for hydrosphere processes.

These solver classes are helper implementations and are not FunctorBase
runtime modules. They are consumed by scientific Functors (for example
RichardsSoilMoisture) and must not be registered directly to the simulator.
"""

from .richards_solver import RichardsNumericalSolver

__all__ = ["RichardsNumericalSolver"]
