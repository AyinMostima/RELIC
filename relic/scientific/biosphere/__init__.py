"""
Biosphere processes for ecosystem modeling.

This package provides comprehensive terrestrial biosphere modeling capabilities
including photosynthesis, respiration, phenology, growth, and BVOC emissions.
"""

# Import submodules
from . import bvoc, growth, pft, phenology, photosynthesis, respiration, stress, surrogates
from .phenology import (
    CropPhenology,
    GDDPhenology,
    GSIPhenology,
    PhenologyFunctor,
    StaticPhenology,
)

# Import main functors
from .photosynthesis import FarquharPhotosynthesis
from .respiration.autotrophic_respiration import AutotrophicRespiration
from .respiration.growth_respiration import GrowthRespiration
from .respiration.maintenance_respiration import MaintenanceRespiration
from .stress import NutrientStress, OzoneStress, ThermalStress, WaterStress

__all__ = [
    # Submodules
    "photosynthesis",
    "respiration",
    "phenology",
    "growth",
    "bvoc",
    "pft",
    "stress",
    "surrogates",
    # Photosynthesis functors
    "FarquharPhotosynthesis",
    # Respiration functors
    "MaintenanceRespiration",
    "GrowthRespiration",
    "AutotrophicRespiration",
    # Phenology functors
    "PhenologyFunctor",
    # Stress functors
    "WaterStress",
    "ThermalStress",
    "NutrientStress",
    "OzoneStress",
]


# Convenience function for creating standard biosphere functors
def create_standard_biosphere_functors(
    pft_name=None,
    *,
    include_growth: bool = False,
    include_bvoc: bool = False,
    include_phenology_growth_diagnostic: bool = False,
    use_surrogate_photosynthesis: bool = False,
    phenology_model_owner: str = "gsi_phenology",
    bvoc_model_owner: str = "megan_bvoc_model",
):
    """
    Create a standard set of biosphere functors with default configurations.

    Parameters:
        pft_name : str, optional
            Plant functional type name for PFT-specific parameters
        include_growth : bool, optional
            Include growth functors in the convenience bundle.
            Default False to avoid accidental overlap with explicit growth modules.
        include_bvoc : bool, optional
            Include BVOC functors in the convenience bundle.
            Default False to avoid forcing experimental BVOC choices.
        include_phenology_growth_diagnostic : bool, optional
            Include the read-only selected-owner growth-activity diagnostic.
            Default False; this never creates another phenology model owner.
        use_surrogate_photosynthesis : bool, optional
            If True, use SimpleCanopyPhotosynthesis instead of FarquharPhotosynthesis.
            This is an explicit opt-in self-contained carbon closure. The standard
            maintenance, growth, autotrophic respiration, NPP, and biomass growth
            closure is excluded from this bundle.
        phenology_model_owner : str, optional
            Compile-time selection of exactly one canonical phenology Functor.
            Supported values are static_phenology, gdd_phenology,
            gsi_phenology, and crop_phenology.
        bvoc_model_owner : str, optional
            Compile-time selection of exactly one BVOC closure when
            include_bvoc is True. Supported values are megan_bvoc_model,
            mechanistic_isoprene_synthesis, and
            mechanistic_monoterpene_synthesis.

    Returns:
        Dict[str, FunctorBase]: Dictionary of initialized biosphere functors
    """
    supported_bvoc_owners = {
        "megan_bvoc_model",
        "mechanistic_isoprene_synthesis",
        "mechanistic_monoterpene_synthesis",
    }
    if bvoc_model_owner not in supported_bvoc_owners:
        raise ValueError(
            "Unsupported BVOC model owner "
            f"'{bvoc_model_owner}'; expected megan_bvoc_model, "
            "mechanistic_isoprene_synthesis, or "
            "mechanistic_monoterpene_synthesis."
        )
    if use_surrogate_photosynthesis and include_growth:
        raise ValueError(
            "surrogate photosynthesis cannot be combined with the standard growth closure"
        )
    if use_surrogate_photosynthesis and include_bvoc:
        raise ValueError(
            "surrogate photosynthesis cannot be combined with the default BVOC closure"
        )

    phenology_diagnostics = PhenologyFunctor(
        config={"canonical_model_owner": phenology_model_owner}
    )

    if use_surrogate_photosynthesis:
        from .surrogates.simple_photosynthesis import SimpleCanopyPhotosynthesis

        photosynthesis_functor = SimpleCanopyPhotosynthesis(name="simple_photosynthesis")
    else:
        photosynthesis_functor = FarquharPhotosynthesis(pft_name=pft_name)

    if phenology_model_owner == "static_phenology":
        phenology_model = StaticPhenology(name=phenology_model_owner)
    elif phenology_model_owner == "gdd_phenology":
        phenology_model = GDDPhenology(name=phenology_model_owner)
    elif phenology_model_owner == "gsi_phenology":
        phenology_model = GSIPhenology(name=phenology_model_owner)
    elif phenology_model_owner == "crop_phenology":
        phenology_model = CropPhenology(name=phenology_model_owner)
    else:
        raise ValueError(
            "Unsupported phenology_model_owner "
            f"'{phenology_model_owner}'; expected static_phenology, "
            "gdd_phenology, gsi_phenology, or crop_phenology."
        )
    functors = {
        "photosynthesis": photosynthesis_functor,
        phenology_model_owner: phenology_model,
        "phenology_diagnostics": phenology_diagnostics,
        "water_stress": WaterStress(),
        "thermal_stress": ThermalStress(),
        "nutrient_stress": NutrientStress(),
        "ozone_stress": OzoneStress(),
    }

    if include_phenology_growth_diagnostic:
        from .growth.phenology_growth import PhenologyGrowth

        functors["phenology_growth"] = PhenologyGrowth(
            phenology_model_owner=phenology_model_owner
        )

    if not use_surrogate_photosynthesis:
        functors["maintenance_respiration"] = MaintenanceRespiration(pft_name=pft_name)
        functors["growth_respiration"] = GrowthRespiration(pft_name=pft_name)
        functors["autotrophic_respiration"] = AutotrophicRespiration(pft_name=pft_name)

    if include_growth:
        from .growth.biomass_allocation import BiomassAllocation

        functors["biomass_allocation"] = BiomassAllocation(
            pft_name=pft_name,
            phenology_model_owner=phenology_model_owner,
            bvoc_carbon_allocation_owner=(
                bvoc_model_owner if include_bvoc else None
            ),
        )

    if include_bvoc:
        if bvoc_model_owner == "megan_bvoc_model":
            from .bvoc.megan_model import MEGANModel

            bvoc_functor = MEGANModel()
        elif bvoc_model_owner == "mechanistic_isoprene_synthesis":
            from .bvoc.synthesis.isoprene import IsopreneSynthesis

            bvoc_functor = IsopreneSynthesis()
        else:
            from .bvoc.synthesis.monoterpene import MonoterpeneSynthesis

            bvoc_functor = MonoterpeneSynthesis()
        functors[bvoc_functor.name] = bvoc_functor

    # Preserve process-owned forcing dependencies and append the selected LAI owner.
    # Component and aggregate respiration dependencies are declared by their owners.
    photosynthesis_dependencies = list(
        functors["photosynthesis"].get_dependencies()
    )
    if phenology_model_owner not in photosynthesis_dependencies:
        photosynthesis_dependencies.append(phenology_model_owner)
    functors["photosynthesis"].set_dependencies(photosynthesis_dependencies)
    if not use_surrogate_photosynthesis:
        functors["growth_respiration"].set_dependencies(
            [photosynthesis_functor.name, "maintenance_respiration"]
        )

    return functors


# Module metadata
__version__ = "1.0.0"
__author__ = "Yuchong Long"
