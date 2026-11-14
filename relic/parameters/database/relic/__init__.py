"""
Native RELIC Parameter Database

This directory contains native RELIC parameters organized by scientific domain.
All parameters include full metadata, units, bounds, and scientific references
following CF-1.8 conventions and FAIR data principles.

Organization:
    biosphere/: Biological processes (photosynthesis, respiration, BVOC, growth)
    hydrosphere/: Water cycle processes (ET, soil moisture, runoff)
    pedosphere/: Soil processes (decomposition, nutrient cycling, carbon)
    atmosphere/: Atmospheric processes (radiation, turbulence, chemistry)
    anthroposphere/: Human impact processes (land use, management, emissions)
    physical/: Physical constants and material properties

Schema Requirements:
    - All parameters must have units following CF-1.8 conventions
    - Physical bounds must be scientifically justified with references
    - Description must explain parameter meaning and context
    - References must include DOI or authoritative source
    - Provenance must track source and version information

Example Parameter Format:
    key: biosphere.photosynthesis.vcmax25
    value: 60.0
    units: umol CO2 m-2 s-1
    dtype: float32
    bounds: [10.0, 200.0]
    category: biological_parameter
    description: Maximum carboxylation rate at 25degC reference temperature
    references:
      - "doi:10.1071/PP9900137"
      - "Farquhar et al. (1980) A biochemical model of photosynthetic CO2 assimilation"
    tags:
      - "learnable"
      - "calibratable"
      - "temperature_sensitive"
    dimension: flux
    requires_grad: true
    provenance:
      source: literature_synthesis
      schema_version: "1.0.0"
      created_at: "2024-01-01T00:00:00Z"
      
References:
    - CF-1.8 Convention for Climate and Forecast Metadata
    - FAIR Data Principles (Findable, Accessible, Interoperable, Reusable)
    - Farquhar et al. (1980) photosynthesis model parameters
    - Guenther et al. (2012) MEGAN model parameters
"""

__version__ = "1.0.0"
__tier__ = "relic"
__description__ = "Native RELIC parameters with full scientific validation"