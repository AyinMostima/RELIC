"""Language-agnostic coupling plugin adapters for RELIC."""

from .host import PluginHost
from .adapter import PluginFunctorAdapter

__all__ = ["PluginHost", "PluginFunctorAdapter"]
