"""Shared BVOC ParameterManager test helper.

The historical BVOC tests imported a top-level helper that was no longer
present. This module provides the same test-facing builder while delegating to
the real RELIC parameter manager for parameters that are not explicitly
overridden by a test.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from relic.parameters import get_global_manager as _real_get_global_manager
from relic.parameters.spec import ParameterSpec


_DEFAULT_VALUES: Dict[str, Any] = {
    "biosphere.bvoc.broadleaf_deciduous_temperate.isoprene": 70.0,
    "biosphere.bvoc.broadleaf_deciduous_temperate.monoterpenes": 1.2,
    "biosphere.bvoc.broadleaf_evergreen_tropical.isoprene": 24.0,
    "biosphere.bvoc.broadleaf_evergreen_tropical.monoterpenes": 2.3,
    "biosphere.bvoc.needleleaf_evergreen_boreal.isoprene": 0.8,
    "biosphere.bvoc.needleleaf_evergreen_boreal.monoterpenes": 8.1,
    "biosphere.bvoc.default.isoprene": 24.0,
    "biosphere.bvoc.default.monoterpenes": 2.3,
    "biosphere.bvoc.default.sesquiterpenes": 0.45,
    "biosphere.bvoc.default.ovocs": 1.8,
    "biosphere.bvoc.emission_factors.isoprene": 24.0,
    "biosphere.bvoc.emission_factors.monoterpenes": 2.3,
    "biosphere.bvoc.emission_factors.sesquiterpenes": 0.45,
    "biosphere.bvoc.emission_factors.ovocs": 1.8,
    "biosphere.bvoc.megan.alpha_light": 0.0027,
    "biosphere.bvoc.megan.cl_light": 1.066,
    "biosphere.bvoc.megan.eopt_isoprene": 1.75,
    "biosphere.bvoc.megan.topt": 313.0,
    "biosphere.bvoc.megan.ct1_isoprene": 95000.0,
    "biosphere.bvoc.megan.ct2": 230000.0,
    "biosphere.bvoc.megan.beta_monoterpenes": 0.09,
    "biosphere.bvoc.megan.beta_sesquiterpenes": 0.17,
    "biosphere.bvoc.megan.hybrid_weight": 0.1,
    "biosphere.bvoc.t_standard": 303.15,
    "biosphere.bvoc.co2_inhibition.c_star": 600.0,
    "biosphere.bvoc.water_stress.wilting_point": -1.5,
    "biosphere.bvoc.water_stress.field_capacity": -0.1,
    "relic.physical.constants.molecular_weights.isoprene": 68.12,
    "biosphere.bvoc.molecular_weights.monoterpenes": 136.24,
    "biosphere.bvoc.molecular_weights.sesquiterpenes": 204.36,
    "biosphere.bvoc.molecular_weights.ovocs": 37.87,
    "biosphere.bvoc.carbon_atoms.isoprene": 5.0,
    "biosphere.bvoc.carbon_atoms.monoterpenes": 10.0,
    "biosphere.bvoc.carbon_atoms.sesquiterpenes": 15.0,
    "biosphere.bvoc.carbon_atoms.ovocs": 1.456,
    "biosphere.bvoc.carbon_fraction.default": 0.882,
    "biosphere.bvoc.carbon_fraction.isoprene": 0.882,
    "biosphere.bvoc.carbon_fraction.monoterpenes": 0.882,
    "biosphere.bvoc.carbon_fraction.sesquiterpenes": 0.882,
    "biosphere.bvoc.carbon_fraction.ovocs": 0.4618,
    "biosphere.bvoc.photosynthetic_coupling.f_carbon_isoprene": 0.0147,
    "biosphere.bvoc.photosynthetic_coupling.f_carbon_monoterpenes": 0.005,
    "biosphere.bvoc.photosynthetic_coupling.f_carbon_sesquiterpenes": 0.001,
    "biosphere.bvoc.photosynthetic_coupling.f_carbon_ovocs": 0.002,
    "biosphere.bvoc.monoterpene.v_max_mts": 1.2,
    "biosphere.bvoc.monoterpene.k_gpp": 15.0,
    "biosphere.bvoc.monoterpene.pool.capacity": 734.0,
    "biosphere.bvoc.monoterpene.pool_initial": 50.0,
    "biosphere.bvoc.monoterpene.pool.k_emit_25": 2.222e-05,
    "biosphere.bvoc.monoterpene.pool.beta_emission": 0.09,
    "biosphere.bvoc.monoterpene.pool.k_degradation": 5.556e-07,
    "biosphere.bvoc.monoterpene.q10_synthase": 2.5,
    "biosphere.bvoc.monoterpene.seasonal.amplitude": 0.5,
    "biosphere.bvoc.monoterpene.seasonal.minimum_factor": 0.1,
    "biosphere.bvoc.monoterpene.seasonal.peak_day": 150.0,
    "biosphere.bvoc.monoterpene.enable_seasonal_variation": False,
    "relic.physical.constants.time.seconds_per_day": 86400.0,
    "relic.physical.constants.time.seconds_per_year": 31557600.0,
    "conversions.micrograms_to_grams": 1.0e-6,
    "validation.numerical.coupling_min": 0.0,
    "validation.numerical.coupling_max": 1.0,
}


_UNITS: Dict[str, str] = {
    "alpha_light": "m2 s umol-1",
    "ct1": "J mol-1",
    "ct2": "J mol-1",
    "topt": "K",
    "t_standard": "K",
    "v_max_mts": "umol m-2 s-1",
    "k_gpp": "umol m-2 s-1",
    "pool.capacity": "umol m-2",
    "pool_initial": "umol m-2",
    "k_emit_25": "s-1",
    "k_degradation": "s-1",
    "seconds_per_day": "s",
    "seconds_per_year": "s",
    "peak_day": "day",
    "beta": "K-1",
    "molecular_weights": "g mol-1",
    "carbon_atoms": "dimensionless",
    "carbon_fraction": "dimensionless",
    "co2": "ppm",
    "water_stress": "MPa",
    "micrograms_to_grams": "g ug-1",
}

_BOUNDS: Dict[str, Tuple[float, float]] = {
    "biosphere.bvoc.default.isoprene": (10.0, 50.0),
    "biosphere.bvoc.megan.alpha_light": (0.001, 0.01),
}


def _infer_units(key: str) -> str:
    if key.endswith(("inhibition_factor", "co2_inhibition_min", "co2_inhibition_max")):
        return "dimensionless"
    if key.endswith(("reference_concentration", "baseline_co2", "k_co2", "c_star")):
        return "ppm"
    for token, units in _UNITS.items():
        if token in key:
            return units
    if key.endswith((".isoprene", ".monoterpenes", ".sesquiterpenes", ".ovocs")):
        return "ug g-1 DW h-1"
    return "dimensionless"


def _units_equivalent(actual: str, expected: str) -> bool:
    """Require exact canonical units in BVOC tests."""
    return actual == expected


class RealisticBVOCParameterManager:
    """Overlay manager used by BVOC tests."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        self._real_manager = _real_get_global_manager()
        self._values: Dict[str, Any] = dict(_DEFAULT_VALUES)
        # Match the canonical ParameterManager trainable-parameter surface.
        # FunctorBase uses this mapping before requesting a tensor value for
        # ParameterSpec entries whose requires_grad flag is enabled.
        self.nn_parameters: Dict[str, torch.nn.Parameter] = {}
        if overrides:
            self._values.update(overrides)
        self.parameters: Dict[str, ParameterSpec] = {
            key: self._make_spec(key, value)
            for key, value in self._values.items()
        }
        self._missing_requests: Dict[str, int] = {}

    @property
    def database(self):
        """The ParameterDatabase of the wrapped real manager.

        Test overrides are an overlay of values; the namespaces they live in and
        every non-overridden key come from the real database, which FunctorBase
        consults (is_declared_namespace) to decide whether a hierarchical name
        is an absolute database key.
        """
        return self._real_manager.database

    def _full_key(self, key: str, namespace: Optional[str] = None) -> str:
        if namespace and not key.startswith(namespace):
            return f"{namespace}.{key}"
        return key

    def _make_spec(self, key: str, value: Any) -> ParameterSpec:
        return ParameterSpec(
            key=key,
            value=value,
            units=_infer_units(key),
            bounds=_BOUNDS.get(key, (-float("inf"), float("inf"))),
            description="BVOC test parameter overlay",
            references=["Guenther et al. (2012). MEGAN2.1."],
            provenance={"source": "test.bvoc_parameter_manager_helper"}, dtype="float32", category="model_parameter",
        )

    def get(
        self,
        key: str,
        default: Any = None,
        namespace: Optional[str] = None,
        units: Optional[str] = None,
        expected_type: Optional[type] = None,
        required: bool = False,
        bounds: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Any:
        full_key = self._full_key(key, namespace)
        if full_key in self._values:
            value = self._values[full_key]
            if units is not None:
                actual_units = self.parameters.get(full_key, self._make_spec(full_key, value)).units
                if not _units_equivalent(str(actual_units), str(units)):
                    raise ValueError(
                        f"Unit conversion failed for {full_key}: {actual_units} to {units}"
                    )
        else:
            try:
                value = self._real_manager.get(
                    full_key,
                    default=default,
                    units=units,
                    expected_type=expected_type,
                    required=required,
                    bounds=bounds,
                    **kwargs,
                )
            except KeyError:
                self._missing_requests[full_key] = self._missing_requests.get(full_key, 0) + 1
                if required and default is None:
                    raise
                value = default

        if bounds is not None and isinstance(value, (int, float)):
            lower, upper = bounds
            if lower is not None:
                value = max(float(value), float(lower))
            if upper is not None:
                value = min(float(value), float(upper))
        if expected_type is not None and value is not None and not isinstance(value, expected_type):
            try:
                value = expected_type(value)
            except (TypeError, ValueError):
                pass
        return value

    def get_parameter(self, key: str) -> Optional[ParameterSpec]:
        full_key = self._full_key(key)
        if full_key in self.parameters:
            return self.parameters[full_key]
        return self._real_manager.get_parameter(full_key)

    def list(self, namespace: Optional[str] = None, **kwargs: Any) -> list[str]:
        del kwargs
        local_keys = list(self.parameters)
        if namespace:
            local_keys = [key for key in local_keys if key.startswith(namespace)]
        try:
            real_keys = self._real_manager.list(namespace=namespace)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError):
            real_keys = []
        return sorted(set(local_keys).union(real_keys))

    def set_value(
        self,
        key: str,
        value: Any,
        units: Optional[str] = None,
        validate: bool = True,
        notify: bool = True,
    ) -> None:
        del units, notify
        if validate and key in _BOUNDS and isinstance(value, (int, float)):
            lower, upper = _BOUNDS[key]
            value = min(max(float(value), float(lower)), float(upper))
        self._values[key] = value
        self.parameters[key] = self._make_spec(key, value)

    def add_change_callback(self, callback: Any) -> None:
        if hasattr(self._real_manager, "add_change_callback"):
            self._real_manager.add_change_callback(callback)

    def remove_change_callback(self, callback: Any) -> None:
        if hasattr(self._real_manager, "remove_change_callback"):
            self._real_manager.remove_change_callback(callback)

    def run_registry_audit(self) -> Dict[str, Any]:
        return {
            "manager": "RealisticBVOCParameterManager",
            "overlay_parameter_count": len(self.parameters),
            "missing_request_count": sum(self._missing_requests.values()),
        }

    def get_missing_requests(self) -> Dict[str, int]:
        return dict(self._missing_requests)


def build_realistic_parameter_manager(
    overrides: Optional[Dict[str, Any]] = None,
) -> RealisticBVOCParameterManager:
    """Build a BVOC parameter manager overlay for tests."""

    return RealisticBVOCParameterManager(overrides)
