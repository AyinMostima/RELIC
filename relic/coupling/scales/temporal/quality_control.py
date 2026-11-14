"""
Temporal Quality Control and Coverage Assessment Module

This module provides comprehensive missing data handling, coverage evaluation,
and quality assessment for temporal scaling operations in the RELIC framework.
Implements robust strategies for dealing with gaps, outliers, and insufficient
coverage in temporal data.

Key Features:
- Coverage threshold evaluation with configurable policies
- Missing data detection and classification
- Gap filling strategies for temporal continuity
- Outlier detection and handling
- Quality flag generation and management
- Uncertainty quantification for interpolated values
- Multi-level quality assessment (L1-L4)

Mathematical Framework:
    Coverage calculation:
    coverage_j = Sigma_i w_{ij} where w_{ij} are time overlap weights
    
    Quality threshold:
    valid_j = coverage_j >= threshold
    
    Gap interpolation:
    phi_gap = f(phi_before, phi_after, t_gap, t_before, t_after)
    
    Uncertainty estimation:
    sigma_interp = sigma_base * (1 + gap_length / reference_scale)

Classes:
    QualityController: Main quality control manager
    CoverageEvaluator: Coverage assessment and validation
    GapFiller: Missing data interpolation and filling
    OutlierDetector: Outlier identification and handling
    QualityFlags: Quality flag management

References:
    - WMO Guidelines for Quality Control of Climate Data
    - GCOS Climate Data Rescue and Management Guidelines
    - CF Conventions Quality Flag Standards
    - IPCC Data Quality Guidelines
"""

import warnings
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor

from ....parameters import get_global_manager
from .core import TimeCoordinate, ValidationError

# Set up logging
logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality assessment levels.

    Cutoffs are the ``coupling.scales.temporal.quality_control.grade_*``
    parameters (current values 0.95/1 sample gap, 0.8, 0.6, 0.4).
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class GapFillingMethod(Enum):
    """Gap filling strategies."""
    LINEAR = "linear"                    # Linear interpolation
    CLIMATOLOGICAL = "climatological"   # Use climatological values
    SEASONAL = "seasonal"               # Seasonal decomposition
    SPLINE = "spline"                   # Cubic spline
    NEAREST = "nearest"                 # Nearest neighbor
    NONE = "none"                       # No gap filling


class OutlierMethod(Enum):
    """Outlier detection methods."""
    ZSCORE = "zscore"              # Z-score based (statistical)
    IQR = "iqr"                    # Interquartile range
    MODIFIED_ZSCORE = "modified_zscore"  # Modified Z-score (robust)
    ISOLATION_FOREST = "isolation_forest"  # Machine learning approach
    NONE = "none"                  # No outlier detection


_COVERAGE_THRESHOLD_KEY = "coupling.scales.temporal.coverage_threshold"
_QUALITY_CONTROL_NAMESPACE = "coupling.scales.temporal.quality_control"

# Enum members with an executable implementation in GapFiller/OutlierDetector.
# SEASONAL, SPLINE, and ISOLATION_FOREST have none and would otherwise leave
# gaps unfilled or outliers undetected without an error.
_IMPLEMENTED_GAP_FILLING_METHODS = frozenset(
    {
        GapFillingMethod.LINEAR,
        GapFillingMethod.NEAREST,
        GapFillingMethod.CLIMATOLOGICAL,
        GapFillingMethod.NONE,
    }
)
_IMPLEMENTED_OUTLIER_METHODS = frozenset(
    {
        OutlierMethod.ZSCORE,
        OutlierMethod.MODIFIED_ZSCORE,
        OutlierMethod.IQR,
        OutlierMethod.NONE,
    }
)
# (QualityConfig field, parameter key, expected Python type); every key is
# read with required=True and carries full metadata in scales_parameters.yaml.
_QUALITY_PARAMETER_KEYS = (
    ("coverage_threshold", _COVERAGE_THRESHOLD_KEY, float),
    ("max_gap_length", f"{_QUALITY_CONTROL_NAMESPACE}.max_gap_length", int),
    ("gap_filling_method", f"{_QUALITY_CONTROL_NAMESPACE}.gap_filling_method", str),
    ("outlier_method", f"{_QUALITY_CONTROL_NAMESPACE}.outlier_method", str),
    ("outlier_threshold", f"{_QUALITY_CONTROL_NAMESPACE}.outlier_threshold", float),
    (
        "outlier_min_valid_samples",
        f"{_QUALITY_CONTROL_NAMESPACE}.outlier_min_valid_samples",
        int,
    ),
    ("iqr_fence_factor", f"{_QUALITY_CONTROL_NAMESPACE}.iqr_fence_factor", float),
    (
        "grade_excellent_min_fraction",
        f"{_QUALITY_CONTROL_NAMESPACE}.grade_excellent_min_fraction",
        float,
    ),
    (
        "grade_excellent_max_gap_length",
        f"{_QUALITY_CONTROL_NAMESPACE}.grade_excellent_max_gap_length",
        int,
    ),
    (
        "grade_good_min_fraction",
        f"{_QUALITY_CONTROL_NAMESPACE}.grade_good_min_fraction",
        float,
    ),
    (
        "grade_acceptable_min_fraction",
        f"{_QUALITY_CONTROL_NAMESPACE}.grade_acceptable_min_fraction",
        float,
    ),
    (
        "grade_poor_min_fraction",
        f"{_QUALITY_CONTROL_NAMESPACE}.grade_poor_min_fraction",
        float,
    ),
)


@dataclass(frozen=True)
class QualityConfig:
    """
    Configuration for quality control operations.

    Every field is required and has no Python default. The canonical values
    live in the parameter database and are loaded with
    :meth:`from_parameter_manager` (keys in ``_QUALITY_PARAMETER_KEYS``):

    - ``coverage_threshold`` <- ``coupling.scales.temporal.coverage_threshold``
      (dimensionless, 0 < value <= 1; shared with TemporalAggregator)
    - ``max_gap_length`` (consecutive missing samples filled by GapFiller and
      allowed for GOOD; integer >= 0)
    - ``gap_filling_method``, ``outlier_method`` (implemented members only)
    - ``outlier_threshold`` (absolute z or modified-z score, > 0)
    - ``outlier_min_valid_samples`` (finite samples needed before outlier
      detection; integer >= 2)
    - ``iqr_fence_factor`` (Tukey fence multiplier k, > 0)
    - ``grade_*_min_fraction`` (grade cutoffs with
      0 < poor < acceptable < good < excellent <= 1) and
      ``grade_excellent_max_gap_length`` (integer, <= max_gap_length)
    """
    coverage_threshold: float
    max_gap_length: int
    gap_filling_method: GapFillingMethod
    outlier_method: OutlierMethod
    outlier_threshold: float
    outlier_min_valid_samples: int
    iqr_fence_factor: float
    grade_excellent_min_fraction: float
    grade_excellent_max_gap_length: int
    grade_good_min_fraction: float
    grade_acceptable_min_fraction: float
    grade_poor_min_fraction: float

    def __post_init__(self) -> None:
        coverage = self._finite_real(self.coverage_threshold, "coverage_threshold")
        if not 0.0 < coverage <= 1.0:
            raise ValueError(
                f"coverage_threshold must be in (0, 1]; received {coverage}"
            )
        object.__setattr__(self, "coverage_threshold", coverage)

        gap_length = self._count(self.max_gap_length, "max_gap_length", minimum=0)

        gap_method = self._enum_member(
            self.gap_filling_method, GapFillingMethod, "gap_filling_method"
        )
        if gap_method not in _IMPLEMENTED_GAP_FILLING_METHODS:
            raise ValueError(
                f"gap_filling_method {gap_method.value!r} has no implementation"
            )
        object.__setattr__(self, "gap_filling_method", gap_method)

        outlier_method = self._enum_member(
            self.outlier_method, OutlierMethod, "outlier_method"
        )
        if outlier_method not in _IMPLEMENTED_OUTLIER_METHODS:
            raise ValueError(
                f"outlier_method {outlier_method.value!r} has no implementation"
            )
        object.__setattr__(self, "outlier_method", outlier_method)

        for name in ("outlier_threshold", "iqr_fence_factor"):
            value = self._finite_real(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive; received {value}")
            object.__setattr__(self, name, value)

        # Standard deviation with Bessel correction and quartiles need at
        # least two samples.
        self._count(
            self.outlier_min_valid_samples, "outlier_min_valid_samples", minimum=2
        )

        excellent_gap = self._count(
            self.grade_excellent_max_gap_length,
            "grade_excellent_max_gap_length",
            minimum=0,
        )
        if excellent_gap > gap_length:
            raise ValueError(
                "grade_excellent_max_gap_length must not exceed max_gap_length; "
                f"received {excellent_gap} > {gap_length}"
            )

        cutoffs = []
        for name in (
            "grade_poor_min_fraction",
            "grade_acceptable_min_fraction",
            "grade_good_min_fraction",
            "grade_excellent_min_fraction",
        ):
            value = self._finite_real(getattr(self, name), name)
            object.__setattr__(self, name, value)
            cutoffs.append(value)
        if not (
            0.0 < cutoffs[0] < cutoffs[1] < cutoffs[2] < cutoffs[3] <= 1.0
        ):
            raise ValueError(
                "grade cutoffs must satisfy 0 < poor < acceptable < good < "
                f"excellent <= 1; received {cutoffs}"
            )

    @staticmethod
    def _finite_real(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real scalar; received {value!r}")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite; received {result}")
        return result

    @staticmethod
    def _count(value: object, name: str, *, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer sample count; received {value!r}"
            )
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}; received {value}")
        return value

    @staticmethod
    def _enum_member(value: object, enum_type: type, name: str):
        if isinstance(value, enum_type):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a {enum_type.__name__} or string; received {value!r}"
            )
        try:
            return enum_type(value.strip().lower())
        except ValueError as exc:
            raise ValueError(f"Unknown {name}: {value!r}") from exc

    @classmethod
    def from_parameter_manager(
        cls,
        parameter_manager: Optional[Any] = None,
    ) -> "QualityConfig":
        """Load every threshold from the parameter database (required=True)."""
        manager = (
            parameter_manager
            if parameter_manager is not None
            else get_global_manager()
        )
        if manager is None:
            raise RuntimeError(
                "ParameterManager is required to build QualityConfig"
            )
        values: Dict[str, Any] = {}
        for field_name, key, expected_type in _QUALITY_PARAMETER_KEYS:
            lookup: Dict[str, Any] = {"expected_type": expected_type, "required": True}
            if expected_type is not str:
                lookup["units"] = "dimensionless"
            values[field_name] = manager.get(key, **lookup)
        return cls(**values)


@dataclass
class QualityMetrics:
    """Quality assessment metrics.

    ``insufficient_coverage_count`` is the number of series whose valid
    fraction is below ``coupling.scales.temporal.coverage_threshold``.
    """
    coverage_fraction: float
    missing_fraction: float
    gap_count: int
    max_gap_length: int
    outlier_count: int
    interpolated_count: int
    quality_level: QualityLevel
    insufficient_coverage_count: int
    uncertainty_mean: float = 0.0
    uncertainty_std: float = 0.0


@dataclass
class QualityFlags:
    """
    Quality flags for temporal data.
    
    Bit-packed flags following CF conventions:
    0: Good data
    1: Missing/insufficient coverage
    2: Interpolated/gap-filled
    3: Outlier detected
    4: Questionable quality
    5: Bad data
    """
    flags: Tensor  # [time_steps] integer flags
    flag_meanings: List[str] = field(default_factory=lambda: [
        'good_data',
        'missing_or_insufficient_coverage', 
        'interpolated_or_gap_filled',
        'outlier_detected',
        'questionable_quality',
        'bad_data'
    ])
    
    def set_flag(self, indices: Tensor, flag_bit: int) -> None:
        """Set specific flag bit for given indices."""
        self.flags[indices] = self.flags[indices] | (1 << flag_bit)
    
    def has_flag(self, indices: Tensor, flag_bit: int) -> Tensor:
        """Check if specific flag bit is set."""
        return (self.flags[indices] & (1 << flag_bit)) > 0
    
    def get_good_data_mask(self) -> Tensor:
        """Get mask for good quality data."""
        return self.flags == 0


class CoverageEvaluator:
    """
    Temporal coverage evaluation and validation.
    
    Assesses data coverage, identifies gaps, and evaluates temporal
    completeness for aggregation and disaggregation operations.
    """
    
    def __init__(self, config: QualityConfig):
        """Initialize coverage evaluator."""
        self.config = config
    
    def evaluate_coverage(
        self,
        data: Tensor,
        time_coord: TimeCoordinate,
        weights: Optional[Tensor] = None
    ) -> Tuple[Tensor, QualityMetrics]:
        """
        Evaluate temporal coverage of data.
        
        Parameters:
            data : Tensor
                Input data tensor [time_steps] or [batch_size, time_steps]
            time_coord : TimeCoordinate
                Time coordinate
            weights : Optional[Tensor]
                Time weights for coverage calculation
                
        Returns:
            Tuple[Tensor, QualityMetrics]: (coverage, metrics)
        """
        # Handle multi-dimensional data
        if data.dim() > 1:
            # Work with flattened data for coverage analysis
            original_shape = data.shape
            data_flat = data.view(-1, original_shape[-1])
            batch_size = data_flat.shape[0]
        else:
            data_flat = data.unsqueeze(0)
            batch_size = 1
        
        n_time = data_flat.shape[1]
        
        # Identify missing data
        missing_mask = torch.isnan(data_flat) | torch.isinf(data_flat)
        valid_mask = ~missing_mask
        
        # Compute coverage for each batch item
        if weights is not None:
            # Weighted coverage
            coverage = torch.sum(valid_mask.float() * weights.unsqueeze(0), dim=1) / torch.sum(weights)
        else:
            # Simple fraction coverage
            coverage = torch.sum(valid_mask.float(), dim=1) / n_time
        
        # Compute aggregate metrics across all batch items
        overall_coverage = float(torch.mean(coverage))
        overall_missing = 1.0 - overall_coverage

        # Gap analysis
        gap_stats = self._analyze_gaps(valid_mask[0] if batch_size == 1 else valid_mask)
        
        # Determine quality level
        quality_level = self._determine_quality_level(overall_coverage, gap_stats)

        # Series-level acceptance against the coverage threshold (warns when
        # any series is insufficient); it does not flag individual samples.
        _accepted, n_insufficient = self.validate_coverage_threshold(coverage)

        metrics = QualityMetrics(
            coverage_fraction=overall_coverage,
            missing_fraction=overall_missing,
            gap_count=gap_stats['gap_count'],
            max_gap_length=gap_stats['max_gap_length'],
            outlier_count=0,  # Will be set by outlier detector
            interpolated_count=0,  # Will be set by gap filler
            quality_level=quality_level,
            insufficient_coverage_count=n_insufficient,
        )
        
        # Reshape coverage back to match input batch dimensions
        if data.dim() > 1:
            coverage = coverage.view(original_shape[:-1])
        else:
            coverage = coverage.squeeze(0)
        
        return coverage, metrics
    
    def _analyze_gaps(self, valid_mask: Tensor) -> Dict[str, int]:
        """Analyze gap structure in data."""
        if valid_mask.dim() > 1:
            # Analyze first batch item as representative
            mask_1d = valid_mask[0]
        else:
            mask_1d = valid_mask
        
        # Find gap runs
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, is_valid in enumerate(mask_1d):
            if not is_valid and not in_gap:
                # Start of new gap
                in_gap = True
                gap_start = i
            elif is_valid and in_gap:
                # End of gap
                gap_length = i - gap_start
                gaps.append(gap_length)
                in_gap = False
        
        # Handle gap at end
        if in_gap:
            gap_length = len(mask_1d) - gap_start
            gaps.append(gap_length)
        
        return {
            'gap_count': len(gaps),
            'max_gap_length': max(gaps) if gaps else 0,
            'total_gap_length': sum(gaps)
        }
    
    def _determine_quality_level(
        self,
        coverage: float,
        gap_stats: Optional[Dict[str, int]],
    ) -> QualityLevel:
        """Grade a valid (or unflagged) sample fraction.

        References: RELIC temporal quality-control grading convention
            (time_scaling.md Section 5.2); cutoffs are parameters, not literals.
        Mathematical expression: level = EXCELLENT if f >= f_E and g <= g_E;
            GOOD if f >= f_G and g <= g_max; ACCEPTABLE if f >= f_A;
            POOR if f >= f_P; otherwise UNACCEPTABLE.
        Parameter meanings: f fraction of valid samples; g longest run of
            missing samples; f_E, f_G, f_A, f_P the grade_*_min_fraction
            cutoffs; g_E grade_excellent_max_gap_length; g_max max_gap_length.
        Units: f and cutoffs dimensionless; g, g_E, g_max sample counts.
        Domain/constraints: 0 < f_P < f_A < f_G < f_E <= 1 and g_E <= g_max
            (enforced by QualityConfig). With gap_stats None the gap criteria
            are not applied (final grade of flagged samples).
        Sign convention: not applicable (classification).
        """
        config = self.config
        max_gap = None if gap_stats is None else gap_stats['max_gap_length']
        if coverage >= config.grade_excellent_min_fraction and (
            max_gap is None or max_gap <= config.grade_excellent_max_gap_length
        ):
            return QualityLevel.EXCELLENT
        if coverage >= config.grade_good_min_fraction and (
            max_gap is None or max_gap <= config.max_gap_length
        ):
            return QualityLevel.GOOD
        if coverage >= config.grade_acceptable_min_fraction:
            return QualityLevel.ACCEPTABLE
        if coverage >= config.grade_poor_min_fraction:
            return QualityLevel.POOR
        return QualityLevel.UNACCEPTABLE
    
    def validate_coverage_threshold(
        self,
        coverage: Tensor,
        raise_on_insufficient: bool = False
    ) -> Tuple[Tensor, int]:
        """
        Validate coverage against threshold.
        
        Parameters:
            coverage : Tensor
                Coverage values
            raise_on_insufficient : bool
                Whether to raise exception on insufficient coverage
                
        Returns:
            Tuple[Tensor, int]: (valid_mask, n_insufficient)
        """
        valid_mask = coverage >= self.config.coverage_threshold
        n_insufficient = int(torch.sum(~valid_mask).item())
        
        if n_insufficient > 0:
            if raise_on_insufficient:
                raise ValidationError(
                    f"{n_insufficient} periods have insufficient coverage "
                    f"(< {self.config.coverage_threshold:.2f})"
                )
            else:
                warnings.warn(
                    f"{n_insufficient} periods have insufficient coverage "
                    f"(< {self.config.coverage_threshold:.2f})"
                )
        
        return valid_mask, n_insufficient


class OutlierDetector:
    """
    Temporal outlier detection for quality control.
    
    Identifies anomalous values in time series data using
    statistical and robust methods.
    """
    
    def __init__(self, config: QualityConfig):
        """Initialize outlier detector."""
        self.config = config
    
    def detect_outliers(self, data: Tensor, time_coord: TimeCoordinate) -> Tensor:
        """
        Detect outliers in temporal data.
        
        Parameters:
            data : Tensor
                Input data
            time_coord : TimeCoordinate
                Time coordinate
                
        Returns:
            Tensor: Boolean mask of outliers
        """
        if self.config.outlier_method == OutlierMethod.NONE:
            return torch.zeros_like(data, dtype=torch.bool)
        
        # Handle multi-dimensional data
        original_shape = data.shape
        if data.dim() > 1:
            data_flat = data.view(-1, original_shape[-1])
        else:
            data_flat = data.unsqueeze(0)
        
        outlier_mask = torch.zeros_like(data_flat, dtype=torch.bool)
        
        for i in range(data_flat.shape[0]):
            series = data_flat[i]
            valid_mask = ~(torch.isnan(series) | torch.isinf(series))
            
            if torch.sum(valid_mask) < self.config.outlier_min_valid_samples:
                continue
                
            valid_data = series[valid_mask]
            
            if self.config.outlier_method == OutlierMethod.ZSCORE:
                outliers = self._zscore_outliers(valid_data)
            elif self.config.outlier_method == OutlierMethod.MODIFIED_ZSCORE:
                outliers = self._modified_zscore_outliers(valid_data)
            elif self.config.outlier_method == OutlierMethod.IQR:
                outliers = self._iqr_outliers(valid_data)
            else:
                outliers = torch.zeros_like(valid_data, dtype=torch.bool)
            
            # Map back to full series
            outlier_mask[i, valid_mask] = outliers
        
        # Reshape back to original shape
        if data.dim() > 1:
            outlier_mask = outlier_mask.view(original_shape)
        else:
            outlier_mask = outlier_mask.squeeze(0)
        
        return outlier_mask
    
    @staticmethod
    def _spread_is_rounding_level(spread: Tensor, data: Tensor) -> bool:
        """Return True when a spread cannot be told apart from rounding.

        References: Higham, N. J. (2002), Accuracy and Stability of Numerical
            Algorithms, 2nd ed., SIAM, Section 2.1 (unit roundoff).
        Mathematical expression: degenerate if s <= eps * max_i |x_i|.
        Parameter meanings: s is the spread (std, MAD, or IQR); eps is the
            machine epsilon of the data dtype; x_i are the valid samples.
        Units: s and max_i |x_i| share the data units; eps is dimensionless.
        Domain/constraints: numerical safety guard only, not a model
            parameter; it scales with the data, so detection does not depend
            on the unit or magnitude of the series (a fixed absolute floor
            would call small-magnitude data constant).
        Sign convention: spreads and magnitudes are nonnegative.
        """
        scale = torch.max(torch.abs(data))
        return bool(spread <= torch.finfo(data.dtype).eps * scale)

    def _zscore_outliers(self, data: Tensor) -> Tensor:
        """Detect outliers using the z-score.

        References: Iglewicz and Hoaglin (1993), How to Detect and Handle
            Outliers, ASQC Quality Press.
        Mathematical expression: z_i = |x_i - mean(x)| / s; outlier if z_i > k.
        Parameter meanings: s sample standard deviation (Bessel corrected);
            k outlier_threshold.
        Units: x and s share the data units; z_i and k are dimensionless.
        Domain/constraints: n >= outlier_min_valid_samples finite samples;
            no outliers when s is at rounding level.
        Sign convention: absolute score, both tails flagged.
        """
        mean = torch.mean(data)
        std = torch.std(data)
        if self._spread_is_rounding_level(std, data):
            return torch.zeros_like(data, dtype=torch.bool)
        z_scores = torch.abs((data - mean) / std)
        return z_scores > self.config.outlier_threshold

    def _modified_zscore_outliers(self, data: Tensor) -> Tensor:
        """Detect outliers using the modified z-score.

        References: Iglewicz and Hoaglin (1993), How to Detect and Handle
            Outliers, ASQC Quality Press (M_i definition with 0.6745).
        Mathematical expression: M_i = 0.6745 (x_i - median(x)) / MAD,
            MAD = median(|x_i - median(x)|); outlier if |M_i| > k.
        Parameter meanings: 0.6745 is the standard normal 0.75 quantile of the
            published definition (makes MAD consistent with the standard
            deviation under normality), not a tunable parameter;
            k outlier_threshold.
        Units: x and MAD share the data units; M_i and k are dimensionless.
        Domain/constraints: n >= outlier_min_valid_samples finite samples;
            no outliers when MAD is at rounding level.
        Sign convention: absolute score, both tails flagged.
        """
        median = torch.median(data)
        mad = torch.median(torch.abs(data - median))
        if self._spread_is_rounding_level(mad, data):
            return torch.zeros_like(data, dtype=torch.bool)
        modified_z_scores = 0.6745 * (data - median) / mad
        return torch.abs(modified_z_scores) > self.config.outlier_threshold

    def _iqr_outliers(self, data: Tensor) -> Tensor:
        """Detect outliers using Tukey fences.

        References: Tukey, J. W. (1977), Exploratory Data Analysis,
            Addison-Wesley.
        Mathematical expression: IQR = Q3 - Q1; outlier if
            x_i < Q1 - c IQR or x_i > Q3 + c IQR.
        Parameter meanings: Q1, Q3 sample quartiles (linear interpolation);
            c iqr_fence_factor.
        Units: x, Q1, Q3, and IQR share the data units; c is dimensionless.
        Domain/constraints: n >= outlier_min_valid_samples finite samples;
            no outliers when IQR is at rounding level.
        Sign convention: both tails flagged.
        """
        q1 = torch.quantile(data, 0.25)
        q3 = torch.quantile(data, 0.75)
        iqr = q3 - q1
        if self._spread_is_rounding_level(iqr, data):
            return torch.zeros_like(data, dtype=torch.bool)
        factor = self.config.iqr_fence_factor
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return (data < lower_bound) | (data > upper_bound)


class GapFiller:
    """
    Temporal gap filling for missing data interpolation.
    
    Provides various strategies for filling gaps in time series
    data while maintaining temporal consistency.
    """
    
    def __init__(self, config: QualityConfig):
        """Initialize gap filler."""
        self.config = config
    
    def fill_gaps(
        self,
        data: Tensor,
        time_coord: TimeCoordinate,
        missing_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Fill gaps in temporal data.
        
        Parameters:
            data : Tensor
                Input data with gaps
            time_coord : TimeCoordinate
                Time coordinate
            missing_mask : Tensor
                Boolean mask of missing values
                
        Returns:
            Tuple[Tensor, Tensor]: (filled_data, interpolated_mask)
        """
        if self.config.gap_filling_method == GapFillingMethod.NONE:
            return data, torch.zeros_like(missing_mask, dtype=torch.bool)
        
        filled_data = data.clone()
        interpolated_mask = torch.zeros_like(missing_mask, dtype=torch.bool)
        
        # Handle multi-dimensional data
        original_shape = data.shape
        if data.dim() > 1:
            data_flat = filled_data.view(-1, original_shape[-1])
            missing_flat = missing_mask.view(-1, original_shape[-1])
        else:
            data_flat = filled_data.unsqueeze(0)
            missing_flat = missing_mask.unsqueeze(0)
        
        for i in range(data_flat.shape[0]):
            series = data_flat[i]
            missing = missing_flat[i]
            
            if not torch.any(missing):
                continue
            
            # Fill gaps for this series
            filled_series, interp_mask = self._fill_series_gaps(
                series, missing, time_coord
            )
            
            data_flat[i] = filled_series
            if data.dim() > 1:
                interp_idx = torch.unravel_index(
                    i * original_shape[-1] + torch.where(interp_mask)[0],
                    original_shape
                )
                interpolated_mask[interp_idx] = True
            else:
                interpolated_mask = interp_mask
        
        # Reshape back to original shape
        if data.dim() > 1:
            filled_data = data_flat.view(original_shape)
        else:
            filled_data = data_flat.squeeze(0)
        
        return filled_data, interpolated_mask
    
    def _fill_series_gaps(
        self,
        series: Tensor,
        missing_mask: Tensor,
        time_coord: TimeCoordinate
    ) -> Tuple[Tensor, Tensor]:
        """Fill gaps in a single time series."""
        filled_series = series.clone()
        interpolated_mask = torch.zeros_like(missing_mask, dtype=torch.bool)
        
        if not torch.any(missing_mask):
            return filled_series, interpolated_mask
        
        # Find gap regions
        gaps = self._find_gap_regions(missing_mask)
        
        for gap_start, gap_end in gaps:
            gap_length = gap_end - gap_start
            
            if gap_length > self.config.max_gap_length:
                logger.warning(f"Gap too long ({gap_length} > {self.config.max_gap_length}), skipping")
                continue
            
            # Fill this gap
            filled_values = self._interpolate_gap(
                series, gap_start, gap_end, time_coord
            )
            
            if filled_values is not None:
                filled_series[gap_start:gap_end] = filled_values
                interpolated_mask[gap_start:gap_end] = True
        
        return filled_series, interpolated_mask
    
    def _find_gap_regions(self, missing_mask: Tensor) -> List[Tuple[int, int]]:
        """Find continuous gap regions."""
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, is_missing in enumerate(missing_mask):
            if is_missing and not in_gap:
                # Start of new gap
                in_gap = True
                gap_start = i
            elif not is_missing and in_gap:
                # End of gap
                gaps.append((gap_start, i))
                in_gap = False
        
        # Handle gap at end
        if in_gap:
            gaps.append((gap_start, len(missing_mask)))
        
        return gaps
    
    def _interpolate_gap(
        self,
        series: Tensor,
        gap_start: int,
        gap_end: int,
        time_coord: TimeCoordinate
    ) -> Optional[Tensor]:
        """Interpolate values for a single gap."""
        gap_end - gap_start
        
        if self.config.gap_filling_method == GapFillingMethod.LINEAR:
            return self._linear_interpolation(series, gap_start, gap_end, time_coord)
        elif self.config.gap_filling_method == GapFillingMethod.NEAREST:
            return self._nearest_interpolation(series, gap_start, gap_end)
        elif self.config.gap_filling_method == GapFillingMethod.CLIMATOLOGICAL:
            return self._climatological_interpolation(series, gap_start, gap_end, time_coord)
        else:
            logger.warning(f"Unsupported gap filling method: {self.config.gap_filling_method}")
            return None
    
    def _linear_interpolation(
        self,
        series: Tensor,
        gap_start: int,
        gap_end: int,
        time_coord: TimeCoordinate
    ) -> Optional[Tensor]:
        """Linear interpolation between valid endpoints."""
        # Find valid values before and after gap
        before_idx = gap_start - 1
        after_idx = gap_end
        
        if before_idx < 0 or after_idx >= len(series):
            return None  # Cannot interpolate at boundaries
        
        if torch.isnan(series[before_idx]) or torch.isnan(series[after_idx]):
            return None  # No valid endpoints
        
        # Linear interpolation
        gap_end - gap_start
        t_before = time_coord.values[before_idx]
        t_after = time_coord.values[after_idx]
        
        # Time points for interpolation
        t_gap = time_coord.values[gap_start:gap_end]
        
        # Linear weights
        alpha = (t_gap - t_before) / (t_after - t_before)
        
        # Interpolated values
        v_before = series[before_idx]
        v_after = series[after_idx]
        
        interpolated = (1 - alpha) * v_before + alpha * v_after
        
        return interpolated
    
    def _nearest_interpolation(
        self,
        series: Tensor,
        gap_start: int,
        gap_end: int
    ) -> Optional[Tensor]:
        """Nearest neighbor interpolation."""
        # Find nearest valid value
        before_idx = gap_start - 1
        after_idx = gap_end
        
        valid_before = (before_idx >= 0 and 
                       not torch.isnan(series[before_idx]))
        valid_after = (after_idx < len(series) and 
                      not torch.isnan(series[after_idx]))
        
        if valid_before and valid_after:
            # Choose closer value (simple midpoint rule)
            gap_center = (gap_start + gap_end) // 2
            if gap_center - gap_start <= gap_end - gap_center:
                fill_value = series[before_idx]
            else:
                fill_value = series[after_idx]
        elif valid_before:
            fill_value = series[before_idx]
        elif valid_after:
            fill_value = series[after_idx]
        else:
            return None
        
        return fill_value.repeat(gap_end - gap_start)
    
    def _climatological_interpolation(
        self,
        series: Tensor,
        gap_start: int,
        gap_end: int,
        time_coord: TimeCoordinate
    ) -> Optional[Tensor]:
        """Climatological value interpolation (simplified)."""
        # For now, use simple mean of valid values
        valid_mask = ~torch.isnan(series)
        if torch.sum(valid_mask) == 0:
            return None
        
        climatological_value = torch.mean(series[valid_mask])
        return climatological_value.repeat(gap_end - gap_start)


class QualityController:
    """
    Main quality control coordinator for temporal data.
    
    Integrates coverage evaluation, outlier detection, gap filling,
    and quality flag management for comprehensive quality control.
    """
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize quality controller.

        Without an explicit ``config`` every threshold is loaded from the
        parameter database through :meth:`QualityConfig.from_parameter_manager`.
        """
        self.config = (
            config if config is not None else QualityConfig.from_parameter_manager()
        )
        
        self.coverage_evaluator = CoverageEvaluator(self.config)
        self.outlier_detector = OutlierDetector(self.config)
        self.gap_filler = GapFiller(self.config)
        
        logger.info(f"Initialized QualityController with coverage_threshold={self.config.coverage_threshold}")
    
    def assess_quality(
        self,
        data: Tensor,
        time_coord: TimeCoordinate,
        weights: Optional[Tensor] = None
    ) -> Tuple[Tensor, QualityFlags, QualityMetrics]:
        """
        Comprehensive quality assessment of temporal data.
        
        Parameters:
            data : Tensor
                Input data to assess
            time_coord : TimeCoordinate
                Time coordinate
            weights : Optional[Tensor]
                Time weights for coverage calculation
                
        Returns:
            Tuple[Tensor, QualityFlags, QualityMetrics]: 
                (processed_data, quality_flags, metrics)
        """
        logger.info("Starting comprehensive quality assessment")
        
        # Initialize quality flags
        quality_flags = QualityFlags(
            flags=torch.zeros(data.shape[-1], dtype=torch.int32)
        )
        
        # 1. Evaluate coverage
        coverage, coverage_metrics = self.coverage_evaluator.evaluate_coverage(
            data, time_coord, weights
        )
        
        # 2. Detect missing data
        missing_mask = torch.isnan(data) | torch.isinf(data)
        if data.dim() > 1:
            # Use first batch as representative for flags
            missing_mask_1d = missing_mask[0] if data.dim() == 2 else missing_mask.view(-1, data.shape[-1])[0]
        else:
            missing_mask_1d = missing_mask
        
        # Set missing data flags
        quality_flags.set_flag(torch.where(missing_mask_1d)[0], 1)
        
        # 3. Detect outliers
        outlier_mask = self.outlier_detector.detect_outliers(data, time_coord)
        if data.dim() > 1:
            outlier_mask_1d = outlier_mask[0] if data.dim() == 2 else outlier_mask.view(-1, data.shape[-1])[0]
        else:
            outlier_mask_1d = outlier_mask
        
        # Set outlier flags
        quality_flags.set_flag(torch.where(outlier_mask_1d)[0], 3)
        coverage_metrics.outlier_count = int(torch.sum(outlier_mask_1d).item())
        
        # 4. Fill gaps if requested
        processed_data = data.clone()
        if self.config.gap_filling_method != GapFillingMethod.NONE:
            processed_data, interpolated_mask = self.gap_filler.fill_gaps(
                processed_data, time_coord, missing_mask
            )
            
            if data.dim() > 1:
                interpolated_mask_1d = interpolated_mask[0] if data.dim() == 2 else interpolated_mask.view(-1, data.shape[-1])[0]
            else:
                interpolated_mask_1d = interpolated_mask
            
            # Set interpolated flags
            quality_flags.set_flag(torch.where(interpolated_mask_1d)[0], 2)
            coverage_metrics.interpolated_count = int(torch.sum(interpolated_mask_1d).item())
        
        # 5. Per-sample flags and the series-level coverage verdict.
        # References: CF Conventions 1.8, Section 3.5 (flag_masks and
        #   flag_meanings); WMO Guide to Climatological Practices (2018) on
        #   coverage requirements for period statistics.
        # Mathematical expression: flag_t = OR_b (c_b,t << b) with c_1,t = 1 iff
        #   x_t is NaN or Inf, c_2,t = 1 iff x_t was gap filled, c_3,t = 1 iff
        #   x_t is an outlier; insufficient_s = [C_s < theta] per series s.
        # Parameter meanings: x_t sample t of the representative (first)
        #   series; C_s valid fraction of series s; theta
        #   coupling.scales.temporal.coverage_threshold.
        # Units: flags are bit masks; C_s and theta are dimensionless.
        # Domain/constraints: insufficient coverage is a property of a series,
        #   not of a sample, so it sets no sample flag (bit 4 'questionable'
        #   is not synthesized); it is reported as
        #   metrics.insufficient_coverage_count (computed in evaluate_coverage)
        #   and makes validate_for_operation fail.
        # Sign convention: not applicable (classification).

        # 6. Update overall metrics
        total_flagged = torch.sum(quality_flags.flags > 0).item()
        good_data_fraction = 1.0 - (total_flagged / data.shape[-1])
        # Same parameter cutoffs as the coverage grade; gaps are already
        # reflected by the per-sample flags, so no gap criterion applies.
        final_quality = self.coverage_evaluator._determine_quality_level(
            good_data_fraction,
            None,
        )
        coverage_metrics.quality_level = final_quality
        
        logger.info(
            f"Quality assessment complete: {final_quality.value} "
            f"(coverage={coverage_metrics.coverage_fraction:.3f}, "
            f"outliers={coverage_metrics.outlier_count}, "
            f"interpolated={coverage_metrics.interpolated_count})"
        )
        
        return processed_data, quality_flags, coverage_metrics
    
    def validate_for_operation(
        self,
        data: Tensor,
        time_coord: TimeCoordinate,
        operation: str
    ) -> bool:
        """
        Validate data quality for specific temporal operations.
        
        Parameters:
            data : Tensor
                Data to validate
            time_coord : TimeCoordinate
                Time coordinate
            operation : str
                Operation type ('aggregation' or 'disaggregation')
                
        Returns:
            bool: Whether data meets quality requirements

        References: RELIC temporal quality-control convention; the coverage
            threshold is the same acceptance threshold TemporalAggregator uses.
        Mathematical expression: valid = (N_insufficient == 0) and
            (level >= level_min(operation)).
        Parameter meanings: N_insufficient series with valid fraction below
            coupling.scales.temporal.coverage_threshold; level the graded
            quality of the unflagged fraction; level_min ACCEPTABLE for
            'aggregation' and GOOD for 'disaggregation'.
        Units: dimensionless counts and ordinal levels.
        Domain/constraints: operation must be 'aggregation' or
            'disaggregation'; any other value raises ValueError.
        Sign convention: not applicable (classification).
        """
        if operation == 'aggregation':
            # Aggregation can tolerate some missing data
            min_quality = QualityLevel.ACCEPTABLE
        elif operation == 'disaggregation':
            # Disaggregation requires higher quality
            min_quality = QualityLevel.GOOD
        else:
            raise ValueError(
                "operation must be 'aggregation' or 'disaggregation'; "
                f"received {operation!r}"
            )

        _, _, metrics = self.assess_quality(data, time_coord)
        if metrics.insufficient_coverage_count > 0:
            return False

        quality_levels = [
            QualityLevel.EXCELLENT,
            QualityLevel.GOOD,
            QualityLevel.ACCEPTABLE,
            QualityLevel.POOR,
            QualityLevel.UNACCEPTABLE
        ]
        
        current_level_idx = quality_levels.index(metrics.quality_level)
        required_level_idx = quality_levels.index(min_quality)
        
        return current_level_idx <= required_level_idx


# Export main classes
__all__ = [
    'QualityController',
    'CoverageEvaluator', 
    'GapFiller',
    'OutlierDetector',
    'QualityFlags',
    'QualityConfig',
    'QualityMetrics',
    'QualityLevel',
    'GapFillingMethod',
    'OutlierMethod'
]
