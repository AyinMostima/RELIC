"""Biogenic volatile organic compound algorithms and public contracts.

``MEGANModel`` is the production Functor for RELIC's reduced instantaneous,
dry-leaf-mass-normalized MEGAN-family closure. It combines ParameterManager
emission potentials with temperature, PPFD, canonical ``leaf_area_index``,
atmospheric CO2, and optional upstream stress factors. Its public atmospheric
species ports use ``mol m-2 s-1``.

This package also exposes response algorithms and emission-factor data. These
components do not collapse every BVOC scientific law into one formula. The
existing ContractRegistry defines variable contracts, while simulator
registration and dependency ownership prevent multiple active writers for the
same production output. Species-specific mechanistic Functors remain distinct
alternative providers and must not run beside a conflicting output owner.

The full MEGAN2.1 canopy-history algorithm remains [N]. In particular, this
package does not claim complete canopy radiation integration, 24 h and 240 h
environmental histories, leaf-age classes, PFT-specific leaf-mass transforms,
or bidirectional canopy exchange.

Example
-------
    import torch
    from relic.scientific.biosphere.bvoc import EmissionPathway, MEGANModel

    bvoc = MEGANModel(emission_pathway=EmissionPathway.STANDARD)
    state = {
        "air_temperature": torch.as_tensor(303.15),
        "ppfd": torch.as_tensor(1200.0),
        "leaf_area_index": torch.as_tensor(4.0),
        "atmospheric_co2_ppm": torch.as_tensor(420.0),
    }
    result = bvoc.forward(state, time=0.0, dt=60.0)
    total_bvoc = result["total_bvoc_emission"].value

``STANDARD`` selects the reduced emission-potential closure,
``PHOTOSYNTHESIS`` selects the empirical carbon-allocation closure, and
``HYBRID`` evaluates their configured weighted combination. These pathway
labels do not create another simulator, scheduler, or state owner.
"""


from .megan_model import (
    MEGANModel,
    CouplingStrategy, 
    EmissionPathway
)

from .emission_factors import (
    EmissionFactorDatabase,
    EmissionFactorEntry,
    PlantFunctionalType,
    CompoundCategory,
    DEFAULT_EMISSION_FACTORS
)

from .responses import (BVOCLightResponse,
    BVOCTemperatureResponse,
    BVOCEnvironmentalResponse, 
    BVOCResponseIntegrator
)

# Parameter management imports
__all__ = [
    # Core MEGAN model
    'MEGANModel',
    'CouplingStrategy',
    'EmissionPathway',
    
    # Emission factors
    'EmissionFactorDatabase',
    'EmissionFactorEntry', 
    'PlantFunctionalType',
    'CompoundCategory',
    'DEFAULT_EMISSION_FACTORS',
    
    # Response functions
    'BVOCLightResponse',
    'BVOCTemperatureResponse',
    'BVOCEnvironmentalResponse',
    'BVOCResponseIntegrator',
    
    # historical consistency removed
]

