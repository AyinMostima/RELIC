"""
Data source definitions for external coupling adapters.

References: CF Conventions 1.8; CMIP6 data request.
Parameters: names are dimensionless enums used across adapter modules.
"""

from enum import Enum


class DataSource(Enum):
    """Enumeration of supported external data sources."""

    WRF = "wrf"
    CESM = "cesm"
    CMIP6 = "cmip6"
    VIC = "vic"
    SWAT = "swat"
    UNKNOWN = "unknown"


__all__ = ["DataSource"]
