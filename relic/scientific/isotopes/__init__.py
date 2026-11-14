"""Public fail-closed API for RELIC stable-isotope candidates.

No stable-isotope scientific runtime is currently enabled. Candidate
configuration and numerical fractionation utilities live in their dedicated
modules, while the sole public assembly entry rejects until every governed
state, route, unit, transaction, and restart contract is implemented.
"""

from __future__ import annotations

from .config import IsotopeConfig, get_default_isotope_config
from .registry import wire_isotope_tracker

__version__ = "1.0.0"
__author__ = "Yuchong Long"

__all__ = [
    "wire_isotope_tracker",
    "IsotopeConfig",
    "get_default_config",
    "get_module_info",
    "__version__",
    "__author__",
]


def get_default_config() -> IsotopeConfig:
    """Return the canonical runtime-disabled candidate configuration."""
    return get_default_isotope_config()


def get_module_info() -> dict:
    """Report candidate blockers without claiming operational capability."""
    config = get_default_isotope_config()
    return {
        "version": __version__,
        "author": __author__,
        "status": "[N] runtime_unbound",
        "runtime_available": False,
        "enabled_isotopes": [],
        "output_formats": [],
        "public_entry": "wire_isotope_tracker",
        "blockers": {
            species.value: species_config.runtime_blocker
            for species, species_config in config.species_configs.items()
        },
    }
