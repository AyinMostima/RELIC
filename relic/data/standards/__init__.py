"""
Data Standards Module

Provides support for scientific data standards and conventions including
CF-1.8 (Climate and Forecast) and CMIP6 standards.
"""

from .cf_conventions import (
    CFConventions,
    CFAttributeSynchronizer
)

__all__ = [
    'CFConventions',
    'CFAttributeSynchronizer'
]