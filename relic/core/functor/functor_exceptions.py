"""Functor exception hierarchy."""


class FunctorException(Exception):
    """Base exception class for functor-related errors."""


class ConfigurationError(FunctorException):
    """Raised when functor configuration is invalid."""


class DependencyError(FunctorException):
    """Raised when dependency resolution fails."""


class ExecutionError(FunctorException):
    """Raised when functor execution fails."""


class ConservationError(FunctorException):
    """Raised when conservation laws are violated."""


__all__ = [
    "FunctorException",
    "ConfigurationError",
    "DependencyError",
    "ExecutionError",
    "ConservationError",
]
