"""Bounded canonical temporal scale-coupling surface.

The public Functors share one duration-overlap core and require explicit
``interval_mean``, ``rate``, ``interval_accumulation``, or
``state_at_boundary`` semantics. Executable aggregation and disaggregation are
bounded to explicit interval/calendar/unit contracts. Calendar-dependent
ratio-only mapping, percentile/categorical aggregation, nonuniform profile
reconstruction, and stochastic disaggregation fail closed until their declared
canonical contracts exist.
"""

from ....core.abstractions import TemporalScalingBase
from .aggregator import (
    AggregationConfig,
    AggregationMethod,
    AggregationResult,
    QualityFlags,
    TemporalAggregator,
)
from .calendar import CalendarHandler
from .core import (
    FREQUENCY_HIERARCHY,
    AggregationError,
    CalendarError,
    CFTimeProcessor,
    DisaggregationError,
    PeriodIndexer,
    TimeCoordinate,
    TimeWeightGenerator,
    ValidationError,
    get_calendar_definitions,
    get_supported_calendar_names,
    update_cell_methods,
)
from .disaggregator import (
    DisaggregationConfig,
    DisaggregationMethod,
    DisaggregationResult,
    TemporalDisaggregator,
)
from .quality_control import (
    CoverageEvaluator,
    GapFiller,
    GapFillingMethod,
    OutlierDetector,
    OutlierMethod,
    QualityConfig,
    QualityController,
    QualityLevel,
    QualityMetrics,
)
from .temporal_scaling import TemporalScaling
from .types import (
    ConservationLevel,
    ScalingDirection,
    TemporalConfig,
    TemporalSemantics,
)

QUALITY_CONTROL_AVAILABLE = True
CORE_INFRASTRUCTURE_AVAILABLE = True

__all__ = [
    "TemporalScalingBase",
    "TemporalConfig",
    "TemporalSemantics",
    "ScalingDirection",
    "ConservationLevel",
    "TemporalScaling",
    "TemporalAggregator",
    "TemporalDisaggregator",
    "AggregationMethod",
    "DisaggregationMethod",
    "AggregationConfig",
    "DisaggregationConfig",
    "AggregationResult",
    "DisaggregationResult",
    "QualityFlags",
    "ValidationError",
    "CalendarError",
    "AggregationError",
    "DisaggregationError",
    "CalendarHandler",
    "QualityController",
    "CoverageEvaluator",
    "GapFiller",
    "OutlierDetector",
    "QualityConfig",
    "QualityMetrics",
    "QualityLevel",
    "GapFillingMethod",
    "OutlierMethod",
    "TimeWeightGenerator",
    "PeriodIndexer",
    "CFTimeProcessor",
    "TimeCoordinate",
    "get_calendar_definitions",
    "get_supported_calendar_names",
    "update_cell_methods",
    "FREQUENCY_HIERARCHY",
    "get_module_status",
]

__version__ = "2.0.0"
__author__ = "Yuchong Long"
__description__ = "Bounded canonical temporal scaling"


def get_module_status() -> dict[str, object]:
    """Return bounded capability status without claiming dataset CF conformance."""
    return {
        "unified_classes": True,
        "quality_control": QUALITY_CONTROL_AVAILABLE,
        "core_infrastructure": CORE_INFRASTRUCTURE_AVAILABLE,
        "cf_validation_scope": "time_coordinate_and_cell_method_subset",
        "dataset_cf_conformance_status": "not_implemented",
        "version": __version__,
    }
