"""Fail-closed isotope service boundary.

The former service-owned isotope pools, event ledger, histories, and direct
transformation methods were removed because StateManager is the only allowed
prognostic-state and transaction owner. Stable-isotope runtime remains [N]
until explicit nuclide component states, complete route ledgers, verified
fractionation parameters, and governed UpdateItems are implemented.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from relic.core.state import StateManager
from .config import IsotopeConfig
from .config.models import Nitrogen15Config


class IsotopeTrackingError(Exception):
    """Raised when the runtime-unbound isotope service is requested."""


class IsotopeTrackerService:
    """Public service type whose current executable contract is fail-closed."""

    parameter_namespace = "isotopes.tracker"

    def __init__(
        self,
        isotope_config: IsotopeConfig | Nitrogen15Config,
        state_manager: StateManager,
        variable_mapper: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reject before object, parameter, StateManager, or private-state effects."""
        _ = isotope_config, state_manager, variable_mapper, device
        raise IsotopeTrackingError(
            "[N] isotope runtime_unbound: no governed StateManager component ledger"
        )


__all__ = ["IsotopeTrackerService", "IsotopeTrackingError"]
