"""Shared type definitions for temporal scaling components.

This module centralizes temporal scaling enums and configuration data so
submodules can import these symbols without importing the package root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, TypeVar

import torch

from ..types import SpatialQuantityBasis


class ScalingDirection(Enum):
    """Direction of temporal scaling."""

    UPSCALE = "upscale"
    DOWNSCALE = "downscale"


class TemporalSemantics(Enum):
    """Explicit time-cell meaning, independent of spatial extensivity."""

    INTERVAL_MEAN = "interval_mean"
    RATE = "rate"
    INTERVAL_ACCUMULATION = "interval_accumulation"
    STATE_AT_BOUNDARY = "state_at_boundary"
    POINT_SAMPLE = "point_sample"
    CATEGORICAL = "categorical"


class ConservationLevel(Enum):
    """Conservation enforcement level."""

    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"
    NONE = "none"

    @classmethod
    def from_value(cls, value: "ConservationLevel | str") -> "ConservationLevel":
        """Normalize one explicit conservation policy value."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                "conservation level must be a ConservationLevel or string"
            )
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"Unknown conservation level: {value!r}") from exc

    @property
    def compute_residual(self) -> bool:
        """Whether the active policy computes declared residual diagnostics."""
        return self is not ConservationLevel.NONE

    @property
    def compare_tolerance(self) -> bool:
        """Whether the active policy compares residuals with a resolved tolerance."""
        return self in {ConservationLevel.MODERATE, ConservationLevel.STRICT}

    @property
    def block_on_violation(self) -> bool:
        """Whether a tolerance violation blocks the operation."""
        return self is ConservationLevel.STRICT


_FREQUENCY_TOKENS = frozenset(
    {
        "sub-hourly",
        "hourly",
        "daily",
        "weekly",
        "monthly",
        "seasonal",
        "annual",
    }
)


_EnumT = TypeVar("_EnumT", bound=Enum)


def _normalize_enum(
    value: _EnumT | str,
    enum_type: type[_EnumT],
    field_name: str,
) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be {enum_type.__name__} or string")
    normalized = value.strip().lower()
    try:
        return enum_type(normalized)
    except ValueError as exc:
        raise ValueError(f"Unknown {field_name}: {value!r}") from exc


def _normalize_frequency(freq: str) -> tuple[str, int]:
    freq_lower = freq.lower()
    match = re.match(r"^(\d+)([a-z-]+)$", freq_lower)
    if match:
        multiplier = int(match.group(1))
        base = match.group(2)
    else:
        multiplier = 1
        base = freq_lower
    if multiplier <= 0:
        raise ValueError(f"Frequency multiplier must be positive, got {multiplier}")

    alias_map = {
        "subhourly": "sub-hourly",
        "sub_hourly": "sub-hourly",
        "sub-hourly": "sub-hourly",
        "minute": "sub-hourly",
        "minutes": "sub-hourly",
        "min": "sub-hourly",
        "mins": "sub-hourly",
        "minutely": "sub-hourly",
        "second": "sub-hourly",
        "seconds": "sub-hourly",
        "sec": "sub-hourly",
        "secs": "sub-hourly",
        "secondly": "sub-hourly",
    }
    base = alias_map.get(base, base)
    if base == "yearly":
        base = "annual"
    return base, multiplier


@dataclass
class TemporalConfig:
    """Configuration for temporal scaling operations.

    The valid-coverage acceptance threshold is not a configuration field: its
    sole owner is the parameter ``coupling.scales.temporal.coverage_threshold``
    read by ``TemporalAggregator``.

    ``spatial_basis`` is the spatial quantity representation of the mapped
    field, owned by ``relic.coupling.scales.types.SpatialQuantityBasis``
    (strings are parsed there, so the ambiguous ``extensive`` raises). The
    temporal law is selected by ``temporal_semantics`` only; the temporal
    operators use the basis solely to reject ``CATEGORICAL`` fields.
    """

    source_frequency: str
    target_frequency: str
    direction: ScalingDirection = ScalingDirection.UPSCALE
    spatial_basis: SpatialQuantityBasis = SpatialQuantityBasis.INTENSIVE
    temporal_semantics: Optional[TemporalSemantics] = None
    conservation_level: ConservationLevel = ConservationLevel.MODERATE
    conservation_level_overrides: Mapping[
        str, ConservationLevel | str
    ] = field(default_factory=dict)
    device: Optional[torch.device] = None

    @classmethod
    def from_resolved_settings(
        cls,
        *,
        source_frequency: str,
        target_frequency: str,
        settings: Mapping[str, Any],
        direction: ScalingDirection | str = ScalingDirection.UPSCALE,
        spatial_basis: SpatialQuantityBasis | str = SpatialQuantityBasis.INTENSIVE,
        device: Optional[torch.device] = None,
    ) -> "TemporalConfig":
        """Build a config from one already resolved temporal settings mapping."""
        if not isinstance(settings, Mapping):
            raise TypeError("settings must be a mapping")
        allowed = {
            "conservation_level",
            "conservation_level_overrides",
            "temporal_semantics",
        }
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(
                "Unknown temporal setting keys: " + ", ".join(sorted(unknown))
            )
        return cls(
            source_frequency=source_frequency,
            target_frequency=target_frequency,
            direction=_normalize_enum(
                direction,
                ScalingDirection,
                "scaling direction",
            ),
            spatial_basis=SpatialQuantityBasis.parse(spatial_basis),
            temporal_semantics=(
                None
                if settings.get("temporal_semantics") is None
                else _normalize_enum(
                    settings["temporal_semantics"],
                    TemporalSemantics,
                    "temporal semantics",
                )
            ),
            conservation_level=ConservationLevel.from_value(
                settings.get(
                    "conservation_level",
                    ConservationLevel.MODERATE,
                )
            ),
            conservation_level_overrides=settings.get(
                "conservation_level_overrides",
                {},
            ),
            device=device,
        )

    def resolve_conservation_level(self, functor_name: str) -> ConservationLevel:
        """Resolve one exact functor identity against immutable overrides."""
        self._validate_functor_name(functor_name)
        return ConservationLevel.from_value(
            self.conservation_level_overrides.get(
                functor_name,
                self.conservation_level,
            )
        )

    @staticmethod
    def _validate_functor_name(functor_name: str) -> None:
        if not isinstance(functor_name, str):
            raise TypeError("functor_name must be a string")
        if not functor_name or functor_name.strip() != functor_name:
            raise ValueError(
                "functor_name must be a non-empty exact identity without outer whitespace"
            )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.direction = _normalize_enum(
            self.direction,
            ScalingDirection,
            "scaling direction",
        )
        self.spatial_basis = SpatialQuantityBasis.parse(self.spatial_basis)
        if self.temporal_semantics is not None:
            self.temporal_semantics = _normalize_enum(
                self.temporal_semantics,
                TemporalSemantics,
                "temporal semantics",
            )
        self.conservation_level = ConservationLevel.from_value(
            self.conservation_level
        )

        source_base, _ = _normalize_frequency(self.source_frequency)
        target_base, _ = _normalize_frequency(self.target_frequency)

        if source_base not in _FREQUENCY_TOKENS:
            raise ValueError(f"Unknown source frequency: {self.source_frequency}")
        if target_base not in _FREQUENCY_TOKENS:
            raise ValueError(f"Unknown target frequency: {self.target_frequency}")

        if not isinstance(self.conservation_level_overrides, Mapping):
            raise TypeError("conservation_level_overrides must be a mapping")
        normalized_overrides: dict[str, ConservationLevel] = {}
        for functor_name, level in self.conservation_level_overrides.items():
            self._validate_functor_name(functor_name)
            normalized_overrides[functor_name] = ConservationLevel.from_value(level)
        self.conservation_level_overrides = MappingProxyType(normalized_overrides)


__all__ = [
    "ScalingDirection",
    "TemporalSemantics",
    "ConservationLevel",
    "TemporalConfig",
]
