"""Time staggering for coupled external components (NUOPC-compatible).

NOTE: This module manages clock negotiation for EXTERNAL coupling
(e.g., atmosphere-ocean component synchronization). It is NOT used by the
core RELICSimulator for internal functor scheduling. Internal multi-rate
execution uses fast/medium/slow execution groups with day/month boundary
detection (see relic.core.simulator.time_ops and execution_loop).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from relic.utils.time_constants import (
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
)


class ClockSyncError(RuntimeError):
    """Raised when clock synchronization fails during negotiation."""


class TimeUnit(Enum):
    """Supported time units for clock configuration."""

    SECONDS = "s"
    MINUTES = "min"
    HOURS = "h"
    DAYS = "d"


@dataclass
class ClockConfig:
    """
    Clock configuration for a coupled component.

    References
    ----------
    ESMF Time Manager concepts for component clocks.

    Parameters
    ----------
    time_step : float
        Base time step value.
    unit : TimeUnit, optional
        Time unit (seconds, minutes, hours, days).
    start_time : float, optional
        Simulation start time in seconds.
    stop_time : float, optional
        Simulation end time in seconds.
    allow_subcycling : bool, optional
        If True, this clock can be called multiple times per negotiated step.
    """

    time_step: float
    unit: TimeUnit = TimeUnit.SECONDS
    start_time: float = 0.0
    stop_time: Optional[float] = None
    allow_subcycling: bool = True

    def to_seconds(self) -> float:
        """Convert time_step to seconds."""
        multipliers = {
            TimeUnit.SECONDS: 1.0,
            TimeUnit.MINUTES: float(SECONDS_PER_MINUTE),
            TimeUnit.HOURS: float(SECONDS_PER_HOUR),
            TimeUnit.DAYS: float(SECONDS_PER_DAY),
        }
        return float(self.time_step) * multipliers[self.unit]


@dataclass
class TimeStagger:
    """
    Time staggering offsets for import and export fields.

    Parameters
    ----------
    lag_seconds : float, optional
        Lag applied to import timestamps.
    lead_seconds : float, optional
        Lead applied to export timestamps.
    """

    lag_seconds: float = 0.0
    lead_seconds: float = 0.0
    _clocks: Dict[str, ClockConfig] = field(default_factory=dict)
    _negotiated_dt: Optional[float] = None

    def shift(self, timestamp: float) -> float:
        """
        Apply staggering offsets to a timestamp.

        Parameters
        ----------
        timestamp : float
            Base timestamp in seconds.

        Returns
        -------
        float
            Shifted timestamp.
        """
        return timestamp + self.lead_seconds - self.lag_seconds

    def register_clock(self, component_id: str, clock: ClockConfig) -> None:
        """
        Register a component clock for negotiation.

        Parameters
        ----------
        component_id : str
            Unique component identifier.
        clock : ClockConfig
            Clock configuration for this component.
        """
        if not component_id:
            raise ClockSyncError("component_id must be non-empty")
        self._clocks[component_id] = clock

    def negotiate(self, strategy: str = "lcm") -> float:
        """
        Negotiate a common time step for all registered clocks.

        References
        ----------
        NUOPC clock negotiation patterns.

        Mathematical Expression
        -----------------------
        strategy = "lcm": dt_common = LCM(dt_1, dt_2, ..., dt_n)
        strategy = "min": dt_common = min(dt_1, dt_2, ..., dt_n)
        strategy = "gcd": dt_common = GCD(dt_1, dt_2, ..., dt_n)

        Parameters
        ----------
        strategy : str, optional
            Negotiation strategy ("lcm", "min", "gcd").

        Returns
        -------
        float
            Negotiated common time step in seconds.

        Raises
        ------
        ClockSyncError
            If negotiation fails.
        """
        if not self._clocks:
            raise ClockSyncError("No clocks registered for negotiation")

        dts = [clock.to_seconds() for clock in self._clocks.values()]
        if any(dt <= 0 for dt in dts):
            raise ClockSyncError("All clock time_step values must be positive")

        if strategy == "min":
            self._negotiated_dt = min(dts)
        elif strategy == "gcd":
            from functools import reduce
            from math import gcd

            dts_ms = [int(dt * 1000) for dt in dts]
            gcd_ms = reduce(gcd, dts_ms)
            self._negotiated_dt = gcd_ms / 1000.0
        elif strategy == "lcm":
            from functools import reduce
            from math import gcd

            def lcm(a: int, b: int) -> int:
                return abs(a * b) // gcd(a, b)

            dts_ms = [int(dt * 1000) for dt in dts]
            lcm_ms = reduce(lcm, dts_ms)
            self._negotiated_dt = lcm_ms / 1000.0
        else:
            raise ClockSyncError(f"Unknown negotiation strategy: {strategy}")

        return float(self._negotiated_dt)

    def get_subcycle_ratio(self, component_id: str) -> int:
        """
        Get the integer alignment ratio between a component clock and the negotiated clock.

        Parameters
        ----------
        component_id : str
            Component identifier.

        Returns
        -------
        int
            Integer ratio >= 1 defined as:

            ratio = max(dt_component, dt_negotiated) / min(dt_component, dt_negotiated)

            This value can be interpreted as:
            - If dt_negotiated >= dt_component (typical for strategy="lcm"):
              ratio is the number of component steps per negotiated step.
            - If dt_component >= dt_negotiated (typical for strategy="gcd" or "min"):
              ratio is the number of negotiated steps per component step.
        """
        if self._negotiated_dt is None:
            raise ClockSyncError("Clock negotiation not performed")
        if component_id not in self._clocks:
            raise ClockSyncError(f"Unknown component: {component_id}")

        component_dt = self._clocks[component_id].to_seconds()
        negotiated_ms = int(round(float(self._negotiated_dt) * 1000.0))
        component_ms = int(round(float(component_dt) * 1000.0))
        if negotiated_ms <= 0 or component_ms <= 0:
            raise ClockSyncError("Clock dt values must be positive")

        big = max(negotiated_ms, component_ms)
        small = min(negotiated_ms, component_ms)
        ratio = big / small
        # Guard against non-integer alignment caused by inconsistent dt definitions.
        if abs(ratio - round(ratio)) > 0.01:
            raise ClockSyncError(
                "Component '{}' dt={}s not aligned with negotiated dt={}s".format(
                    component_id, float(component_dt), float(self._negotiated_dt)
                )
            )
        return int(round(ratio))

    def validate_temporal_consistency(self) -> List[str]:
        """
        Validate that all clocks can be synchronized.

        Returns
        -------
        list of str
            Validation warnings (empty if all OK).
        """
        warnings: List[str] = []
        if self._negotiated_dt is None:
            warnings.append("Clock negotiation not performed")
            return warnings

        negotiated_ms = int(round(float(self._negotiated_dt) * 1000.0))
        if negotiated_ms <= 0:
            warnings.append("Negotiated dt must be positive")
            return warnings

        for comp_id, clock in self._clocks.items():
            dt = clock.to_seconds()
            component_ms = int(round(float(dt) * 1000.0))
            if component_ms <= 0:
                warnings.append(
                    "Component '{}' has non-positive dt={}s".format(comp_id, float(dt))
                )
                continue

            big = max(component_ms, negotiated_ms)
            small = min(component_ms, negotiated_ms)
            ratio = big / small
            if abs(ratio - round(ratio)) > 0.01:
                warnings.append(
                    "Component '{}' dt={}s not aligned with negotiated dt={}s".format(
                        comp_id, float(dt), float(self._negotiated_dt)
                    )
                )
        return warnings
