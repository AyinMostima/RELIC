"""Removed compatibility wrappers for parameter access.

Canonical entry points:
1) Functor code: self.get_parameter_value(...)
2) Non-Functor code: get_global_manager().get(...)
"""

from relic.parameters import get_global_manager

__all__ = ["get_global_manager"]
