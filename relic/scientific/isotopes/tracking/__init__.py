"""
Isotope Tracking Services

This package exposes only the fail-closed isotope service boundary.

Available Components:
- IsotopeTrackerService: runtime-unbound service boundary

Key Features:
- No private prognostic pool or event ledger is public.
"""

# Import from existing tracker service module
from ..tracker_service import IsotopeTrackerService

__all__ = [
    'IsotopeTrackerService',
]
