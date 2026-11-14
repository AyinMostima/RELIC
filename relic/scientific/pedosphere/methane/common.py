"""Shared utilities for pedosphere methane functors."""

from __future__ import annotations

import logging

from relic.utils.physical_constants import get_physical_constant

LOGGER = logging.getLogger(__name__)

def fetch_methane_molar_mass() -> float:
    """Return methane molar mass [g mol-1] from the parameter system.

    Parameters
    ----------
    Returns
    -------
        float
        Methane molar mass in grams per mole.
    """
    # References:
    #   - IUPAC molar mass data for methane (CH4)
    # Mathematical expression:
    #   M_CH4 = 16.04 g mol-1 (retrieved from SSOT physical constants database)
    # Parameter meanings:
    #   relic.physical.constants.molecular_weights.ch4: methane molar mass [g mol-1]
    try:
        return float(
            get_physical_constant("relic.physical.constants.molecular_weights.ch4")
        )
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # pragma: no cover - defensive logging
        LOGGER.debug("Failed to read methane molar mass from physical constants: %s", exc)
        raise ValueError(
            "Methane molar mass parameter 'relic.physical.constants.molecular_weights.ch4' "
            "is not available via physical constants SSOT."
        ) from exc


__all__ = ["fetch_methane_molar_mass"]
