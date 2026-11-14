"""
Surrogate biosphere process models.

This namespace hosts experimental or reduced-order process functors that are
not part of the default production biosphere bundle.
"""

from .simple_photosynthesis import SimpleCanopyPhotosynthesis

__all__ = ["SimpleCanopyPhotosynthesis"]
