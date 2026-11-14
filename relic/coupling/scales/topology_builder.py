"""
Topology builder for scale coupling.

Provides reusable helpers to construct and validate hierarchical topologies while
keeping all parameter lookups inside the ParameterManager pipeline. Alias scale pairs
have no synthetic topology: they need an explicit topology with declared areas.

References:
    - RHESSys regional aggregation (Band et al.)
    - CF 1.8 cell_methods for temporal grouping

Mathematical expressions:
    - Area-weighted parent mapping: A_parent = sum(A_child)

Parameter meanings:
    - coupling.scales.hierarchy.*: defines level ordering and unit counts
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Callable, List

from relic.parameters.namespaces import ParameterNamespace
from relic.parameters import UnifiedParameterManager


class TopologyBuilder:
    """Encapsulate hierarchy configuration and topology construction."""

    def __init__(
        self,
        parameter_manager: UnifiedParameterManager,
        get_param: Callable[..., Any],
    ) -> None:
        self.parameter_manager = parameter_manager
        self._get_param = get_param
        self._hierarchy_config: Optional[Dict[str, Any]] = None

    # ------------------------------- Public API -------------------------------
    def validate_topology(self, topology: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """Strictly validate required fields for a topology definition."""
        required_keys = {"n_source_units", "n_target_units", "parent_map", "source_areas", "target_areas"}
        missing = required_keys - set(topology.keys())
        if missing:
            raise ValueError(
                f"Incomplete topology for {operation}: missing keys {missing}. "
                "Configure coupling.scales.hierarchy.* parameters or provide full topology."
            )
        n_src = int(topology["n_source_units"])
        n_tgt = int(topology["n_target_units"])
        parent_map = topology["parent_map"]
        source_areas = topology["source_areas"]
        target_areas = topology["target_areas"]
        if len(source_areas) != n_src:
            raise ValueError(f"source_areas length {len(source_areas)} does not match n_source_units {n_src}")
        if len(target_areas) != n_tgt:
            raise ValueError(f"target_areas length {len(target_areas)} does not match n_target_units {n_tgt}")
        if len(parent_map) != n_src:
            raise ValueError(
                f"parent_map length {len(parent_map)} does not match n_source_units {n_src} for {operation}"
            )
        for child, parent in parent_map.items():
            if child < 0 or child >= n_src:
                raise ValueError(f"parent_map child index {child} out of range for n_source_units={n_src}")
            if parent < 0 or parent >= n_tgt:
                raise ValueError(f"parent_map target index {parent} out of range for n_target_units={n_tgt}")
        return topology

    def ensure_hierarchy_config(self) -> Dict[str, Any]:
        """Lazy-load hierarchy configuration from ParameterManager."""
        if self._hierarchy_config is not None:
            return self._hierarchy_config
        self._hierarchy_config = self._load_hierarchy_parameters(self.parameter_manager)
        return self._hierarchy_config

    def resolve_units_per_target(self, source_scale: str, target_scale: str) -> Optional[int]:
        """Resolve units_per_target based on configured hierarchy levels."""
        hierarchy_config = self.ensure_hierarchy_config()
        levels = hierarchy_config["levels"]
        if source_scale not in levels or target_scale not in levels:
            return None
        source_idx = levels.index(source_scale)
        target_idx = levels.index(target_scale)
        if target_idx <= source_idx:
            return None
        multiplier_keys = {
            ("individual", "patch"): "individual_units_per_patch",
            ("patch", "hillslope"): "patch_units_per_hillslope",
            ("hillslope", "basin"): "hillslope_units_per_basin",
            ("basin", "landscape"): "basin_units_per_landscape",
            ("landscape", "regional"): "landscape_units_per_regional",
        }
        units_per_target = 1
        for idx in range(source_idx, target_idx):
            step = (levels[idx], levels[idx + 1])
            step_key = multiplier_keys.get(step)
            if step_key is None:
                return None
            step_units = hierarchy_config.get(step_key)
            if step_units is None:
                return None
            units_per_target *= step_units
        return units_per_target

    def base_area_for_scale(self, scale: str) -> float:
        """Compute base area for a given hierarchy level using parameter ratios."""
        hierarchy_config = self.ensure_hierarchy_config()
        patch_area = float(hierarchy_config["patch_area"])
        if scale == "individual":
            individuals_per_patch = float(hierarchy_config["individual_units_per_patch"])
            return patch_area / individuals_per_patch
        if scale == "patch":
            return patch_area
        if scale == "hillslope":
            return patch_area * float(hierarchy_config["patch_units_per_hillslope"])
        if scale == "basin":
            return patch_area * float(hierarchy_config["patch_units_per_hillslope"] * hierarchy_config["hillslope_units_per_basin"])
        if scale == "landscape":
            return patch_area * float(
                hierarchy_config["patch_units_per_hillslope"] * hierarchy_config["hillslope_units_per_basin"] * hierarchy_config["basin_units_per_landscape"]
            )
        if scale == "regional":
            return patch_area * float(
                hierarchy_config["patch_units_per_hillslope"]
                * hierarchy_config["hillslope_units_per_basin"]
                * hierarchy_config["basin_units_per_landscape"]
                * hierarchy_config["landscape_units_per_regional"]
            )
        return patch_area

    def build_source_areas(self, n_source_units: int, base_area: float) -> List[float]:
        """Build a list of source areas given a base area with optional variation."""
        if n_source_units <= 0:
            raise ValueError("Number of source units must be positive for topology construction")
        hierarchy_config = self.ensure_hierarchy_config()
        variation = float(hierarchy_config["area_variation_factor"])
        if n_source_units == 1 or variation <= 0.0:
            return [float(base_area) for _ in range(n_source_units)]

        midpoint = (n_source_units - 1) / 2 if n_source_units > 1 else 0.0
        areas: List[float] = []
        for idx in range(n_source_units):
            offset = (idx - midpoint) / max(midpoint, 1.0)
            area = base_area * (1.0 + variation * offset)
            areas.append(max(area, 1e-12))
        return areas

    def infer_default_source_units(self, source_scale: str, target_scale: str) -> int:
        """Infer default source unit count from hierarchy parameters."""
        units_per_target = self.resolve_units_per_target(source_scale, target_scale)
        if units_per_target is None:
            raise RuntimeError(
                f"Cannot infer units_per_target for {source_scale} -> {target_scale}; check hierarchy parameters."
            )
        return max(1, int(units_per_target))

    def compute_child_units(self, coarse_scale: str, fine_scale: str, parent_units: int) -> Optional[int]:
        """Compute number of fine-scale units given coarse unit count."""
        units_per_parent = self.resolve_units_per_target(fine_scale, coarse_scale)
        if units_per_parent is None:
            return None
        return max(1, int(parent_units * units_per_parent))

    def build_balanced_topology(self, n_source_units: int, n_target_units: int) -> Dict[str, Any]:
        """Construct a balanced parent map when detailed topology is unavailable."""
        n_target_units = max(1, n_target_units)
        parent_map: Dict[int, int] = {}
        source_areas: List[float] = []
        target_areas: List[float] = [0.0 for _ in range(n_target_units)]
        for idx in range(n_source_units):
            target_idx = idx % n_target_units
            parent_map[idx] = target_idx
            source_areas.append(1.0)
            target_areas[target_idx] += 1.0
        return {
            "n_source_units": n_source_units,
            "n_target_units": n_target_units,
            "parent_map": parent_map,
            "source_areas": source_areas,
            "target_areas": target_areas,
        }

    def build_topology_from_parameters(
        self,
        source_scale: str,
        target_scale: str,
        n_source_units: int,
        n_target_units: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build topology from hierarchy configuration; returns None if not enough info."""
        hierarchy_config = self.ensure_hierarchy_config()
        levels = hierarchy_config["levels"]
        if not levels or source_scale not in levels or target_scale not in levels:
            return None
        source_idx = levels.index(source_scale)
        target_idx = levels.index(target_scale)
        if target_idx <= source_idx:
            return None
        # Strategy: use configured units_per_target to synthesize topology when explicit maps missing.
        units_per_target = self.resolve_units_per_target(source_scale, target_scale)
        if units_per_target is None:
            return None
        target_units = n_target_units or max(1, math.ceil(n_source_units / units_per_target))
        base_area = self.base_area_for_scale(source_scale)
        source_areas = self.build_source_areas(n_source_units, base_area)
        parent_map: Dict[int, int] = {}
        target_areas: List[float] = [0.0 for _ in range(target_units)]
        for idx, area in enumerate(source_areas):
            target_idx = min(idx // units_per_target, target_units - 1)
            parent_map[idx] = target_idx
            target_areas[target_idx] += area
        return {
            "n_source_units": n_source_units,
            "n_target_units": target_units,
            "parent_map": parent_map,
            "source_areas": source_areas,
            "target_areas": target_areas,
        }

    # ------------------------------ Internal -------------------------------
    def _load_hierarchy_parameters(self, manager: UnifiedParameterManager) -> Dict[str, Any]:
        """Load hierarchy configuration from ParameterManager."""
        namespace = ParameterNamespace.COUPLING_SCALES_HIERARCHY.value
        yaml_hint = "relic\\parameters\\database\\relic\\coupling\\hierarchy_mapping.yaml"

        def fetch_required(key: str, units: str, expected_type: Optional[type]) -> Any:
            """Fetch required parameter with clear error message."""
            namespaced_key = f"{namespace}.{key}"
            try:
                return self._get_param(
                    key,
                    namespace=namespace,
                    units=units,
                    expected_type=expected_type,
                    required=True,
                    validate=True,
                )
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Missing required hierarchy parameter '{namespaced_key}'. "
                    f"Define it in {yaml_hint}."
                ) from exc

        levels = fetch_required("levels", "dimensionless", list)
        if not levels:
            raise ValueError(
                f"Hierarchy levels cannot be empty. Configure '{namespace}.levels' in {yaml_hint}."
            )
        default_source_level = str(fetch_required("default_source_level", "dimensionless", str))
        default_target_level = str(fetch_required("default_target_level", "dimensionless", str))
        if default_source_level not in levels:
            raise ValueError(
                f"default_source_level '{default_source_level}' not found in hierarchy levels {levels}. "
                f"Update {yaml_hint} to keep defaults consistent."
            )
        if default_target_level not in levels:
            raise ValueError(
                f"default_target_level '{default_target_level}' not found in hierarchy levels {levels}. "
                f"Update {yaml_hint} to keep defaults consistent."
            )

        patch_units_per_hillslope = int(fetch_required("patch.units_per_hillslope", "dimensionless", int))
        hillslope_units_per_basin = int(fetch_required("hillslope.units_per_basin", "dimensionless", int))
        basin_units_per_landscape = int(fetch_required("basin.units_per_landscape", "dimensionless", int))
        landscape_units_per_regional = int(fetch_required("landscape.units_per_regional", "dimensionless", int))
        individual_units_per_patch = int(fetch_required("individual.units_per_patch", "dimensionless", int))
        patch_area = float(fetch_required("patch.default_area", "m2", float))
        area_variation_factor = float(fetch_required("area_variation_factor", "dimensionless", float))
        extensive_strategy = str(fetch_required("extensive_variable_strategy", "dimensionless", str))
        intensive_strategy = str(fetch_required("intensive_variable_strategy", "dimensionless", str))
        categorical_strategy = str(fetch_required("categorical_variable_strategy", "dimensionless", str))

        for path, value in {
            "patch.units_per_hillslope": patch_units_per_hillslope,
            "hillslope.units_per_basin": hillslope_units_per_basin,
            "basin.units_per_landscape": basin_units_per_landscape,
            "landscape.units_per_regional": landscape_units_per_regional,
            "individual.units_per_patch": individual_units_per_patch,
        }.items():
            if value <= 0:
                raise ValueError(
                    f"Hierarchy parameter '{namespace}.{path}' must be positive; "
                    f"update {yaml_hint} with a value greater than zero."
                )

        return {
            "levels": list(levels),
            "default_source_level": default_source_level,
            "default_target_level": default_target_level,
            "patch_units_per_hillslope": patch_units_per_hillslope,
            "hillslope_units_per_basin": hillslope_units_per_basin,
            "basin_units_per_landscape": basin_units_per_landscape,
            "landscape_units_per_regional": landscape_units_per_regional,
            "patch_area": patch_area,
            "individual_units_per_patch": individual_units_per_patch,
            "area_variation_factor": area_variation_factor,
            "extensive_strategy": extensive_strategy,
            "intensive_strategy": intensive_strategy,
            "categorical_strategy": categorical_strategy,
        }
