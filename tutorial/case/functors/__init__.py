"""Case-local adapter functors for the Tapajos BVOC tutorial."""

from .case_relic_wrappers import (
    CaseFarquharPhotosynthesis,
    CaseMEGANModel,
    CaseOzoneStress,
    CaseSurfaceRunoff,
    CaseThermalStress,
)
from .infiltration_bridge import InfiltrationBridge
from .irrigation_precip_adapter import IrrigationPrecipAdapter, PrecipThroughfallAdapter
from .ml_surrogates import (
    CaseMEGANDiurnalActivitySurrogate,
    CaseIsopreneResidualSurrogate,
    CaseOzoneStressSurrogate,
    CaseSoilMoistureSurrogate,
    CaseStomatalSurrogate,
    CaseWaterStressSurrogate,
)
from .stomatal_proxy import StomatalProxyFunctor
from .stress_key_adapter import StressKeyAdapter

__all__ = [
    "PrecipThroughfallAdapter",
    "IrrigationPrecipAdapter",
    "InfiltrationBridge",
    "StressKeyAdapter",
    "CaseSoilMoistureSurrogate",
    "CaseWaterStressSurrogate",
    "CaseStomatalSurrogate",
    "CaseOzoneStressSurrogate",
    "CaseMEGANDiurnalActivitySurrogate",
    "CaseIsopreneResidualSurrogate",
    "StomatalProxyFunctor",
    "CaseSurfaceRunoff",
    "CaseThermalStress",
    "CaseOzoneStress",
    "CaseFarquharPhotosynthesis",
    "CaseMEGANModel",
]
