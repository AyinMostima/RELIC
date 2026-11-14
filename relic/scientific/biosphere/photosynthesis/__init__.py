"""
Photosynthesis Processes for RELIC

This module provides unified photosynthesis modeling capabilities for the RELIC 
framework, implementing the Farquhar-von Caemmerer-Berry biochemical model of 
C3 photosynthesis.

Example Usage
-------------
    from relic.scientific.biosphere.photosynthesis import FarquharPhotosynthesis

    # Create photosynthesis model
    photo = FarquharPhotosynthesis(name="farquhar_photosynthesis")
    
    # Prepare input state with required environmental conditions.
    # All inputs should be torch.Tensor for consistency.
    # Required canonical inputs are produced by external forcing,
    # EnvironmentalForcingFunctor, AtmosphericRadiation, and stress owners.
    # The scaling and stress controls are optional (an absent control is the
    # identity 1; jmax_scaling follows vcmax_scaling unless supplied).
    state = {
        "air_temperature": ...,
        "atmospheric_co2_ppm": ...,
        "ppfd": ...,
        "vapor_pressure_deficit": ...,
        "leaf_area_index": ...,
        "vcmax_scaling": ...,
        "jmax_scaling": ...,
        "water_stress_factor": ...,
        "temperature_stress_factor": ...,
        "ozone_stress_factor": ...,
    }

    # Run photosynthesis calculation
    result = photo.forward(state, time=0.0, dt=1.0)

    # Outputs are explicit REPLACE UpdateItems.
    gross = result["gross_photosynthesis"].value
    gpp = result["gross_primary_productivity"].value
"""

from .farquhar import FarquharPhotosynthesis

# Parameter management imports
__all__ = [
    'FarquharPhotosynthesis'
]
