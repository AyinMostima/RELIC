from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from typing import Any, Optional, Tuple, List

import torch
from torch import Tensor

from ..types import ParameterValue, ValidationContext, ValidationLevel
from ..spec import ParameterSpec
from ..ops.compare import values_differ
from ..ops.key_normalization import normalize_parameter_key
from ..ops.unit_helpers import normalize_unit_string
from .nn_params import bind_learnable_value
from .type_utils import caller_owned_value
from .unit_validation import _is_dimensionless_equivalent
from ...data.standards.unit_converter import unit_converter, ConversionContext

logger = logging.getLogger(__name__)


def _bounds_policy_is_error(policy: Any) -> bool:
    if policy is None:
        return False
    value = getattr(policy, "value", policy)
    if isinstance(value, str):
        text = value.lower()
        if text == "error" or text.endswith(".error"):
            return True
    return False


def _get_spec_for_set_value(manager: Any, canonical_key: str) -> ParameterSpec:
    """The specification a value is set on: a copy of the loaded or database one.

    An unknown key raises KeyError: a misspelled key is never turned into a new
    parameter. New parameters are registered explicitly (register_parameter).
    """
    existing_spec = manager.parameters.get(canonical_key)
    if existing_spec is None:
        existing_spec = manager.get_parameter(canonical_key)
    if existing_spec is None:
        raise KeyError(
            f"Unknown parameter '{canonical_key}': register it with register_parameter() "
            "before setting its value"
        )
    return replace(existing_spec)


def _split_validation_errors(result: Any) -> Tuple[List[str], List[str], List[str]]:
    error_types = getattr(result, "error_types", [])
    if error_types:
        bounds_errors = [err for err, et in zip(result.errors, error_types) if et == "bounds"]
        unit_errors = [err for err, et in zip(result.errors, error_types) if et == "units"]
        other_errors = [
            err
            for idx, err in enumerate(result.errors)
            if (idx >= len(error_types)) or (error_types[idx] not in {"bounds", "units"})
        ]
        return bounds_errors, unit_errors, other_errors

    errors_lower = [err.lower() for err in result.errors]
    bounds_errors = [
        err for err, low in zip(result.errors, errors_lower)
        if ("bound" in low or "maximum" in low or "minimum" in low)
    ]
    unit_errors = [
        err for err, low in zip(result.errors, errors_lower) if "unit" in low
    ]
    other_errors = [
        err for err in result.errors if err not in bounds_errors and err not in unit_errors
    ]
    return bounds_errors, unit_errors, other_errors


def _validate_and_resolve_value(
    manager: Any,
    key: str,
    canonical_key: str,
    spec: ParameterSpec,
    initial_value: ParameterValue,
    effective_policy: Any,
) -> Tuple[ParameterValue, bool, Optional[float]]:
    validation_context = ValidationContext(
        all_parameters=manager.storage.get_all(),
        strict_mode=(manager.validation_level == ValidationLevel.STRICT),
    )
    result = manager.validator.validate_value(
        key, initial_value, spec, manager.validation_level, validation_context
    )
    final_value = initial_value
    clamped = False
    clamped_value: Optional[float] = None

    if result.is_valid:
        return final_value, clamped, clamped_value

    bounds_errors, unit_errors, other_errors = _split_validation_errors(result)
    should_error_on_bounds = (
        _bounds_policy_is_error(effective_policy)
        or getattr(manager, "enforce_bounds_error", False)
    )

    if bounds_errors:
        if should_error_on_bounds:
            reason = (
                "strict bounds policy"
                if _bounds_policy_is_error(effective_policy)
                else "enforce_bounds_error"
            )
            raise ValueError(
                f"Parameter validation failed for {key}: {result.errors} ({reason})"
            )
        clamped, clamped_value = manager._clamp_value_to_bounds(spec, final_value)
        if clamped:
            clamp_warning = (
                f"Parameter '{canonical_key}' was clamped to {clamped_value} "
                f"within bounds {getattr(spec, 'bounds', None)}"
            )
            final_value = clamped_value
            result = manager.validator.validate_value(
                key, final_value, spec, manager.validation_level, validation_context
            )
            if hasattr(result, "warnings") and clamp_warning not in result.warnings:
                result.warnings.append(clamp_warning)
            bounds_errors, unit_errors, other_errors = _split_validation_errors(result)

    if other_errors:
        raise ValueError(f"Parameter validation failed for {key}: {result.errors}")
    if bounds_errors and not clamped:
        raise ValueError(f"Parameter validation failed for {key}: {result.errors}")
    if unit_errors and not bounds_errors and not other_errors:
        if getattr(manager, "validation_level", None) == ValidationLevel.NONE:
            warnings.warn(
                f"Parameter '{canonical_key}' uses unrecognized units '{spec.units}'. "
                "Skipping unit validation because validation_level=none.",
                UserWarning,
                stacklevel=3,
            )
        else:
            raise ValueError(
                f"Parameter '{canonical_key}' unit validation failed: {unit_errors}"
            )

    return final_value, clamped, clamped_value


def _apply_post_validation_clamp(
    manager: Any,
    spec: ParameterSpec,
    final_value: ParameterValue,
    validate: bool,
    effective_policy: Any,
    clamped: bool,
    clamped_value: Optional[float],
) -> Tuple[ParameterValue, bool, Optional[float]]:
    if (
        validate
        and getattr(spec, "bounds", None) is not None
        and not _bounds_policy_is_error(effective_policy)
        and not clamped
    ):
        clamp_applied, clamp_value = manager._clamp_value_to_bounds(spec, final_value)
        if clamp_applied:
            clamped = True
            clamped_value = clamp_value
            final_value = clamp_value
    return final_value, clamped, clamped_value


def _raise_if_clamped_with_strict_bounds_policy(
    manager: Any,
    canonical_key: str,
    original_value: ParameterValue,
    spec: ParameterSpec,
    validate: bool,
    effective_policy: Any,
    clamped: bool,
) -> None:
    if (
        clamped
        and validate
        and getattr(manager, "validation_level", None) != ValidationLevel.NONE
        and (
            _bounds_policy_is_error(effective_policy)
            or getattr(manager, "enforce_bounds_error", False)
        )
    ):
        raise ValueError(
            f"Parameter '{canonical_key}' value {original_value} outside bounds {getattr(spec, 'bounds', None)}"
        )


def set_parameter(manager: Any, spec: ParameterSpec, tier: str = "calibrated") -> None:
    """
    Persist one parameter specification and set that parameter in the manager.

    The specification is written to the database tier, then the three owners of
    that one parameter are set together; no other specification is loaded:

    - the specification and the stored value;
    - for a learnable specification, the live binding: an existing binding of
      the same shape and dtype keeps its Parameter object and gradient flag and
      takes the new value, otherwise a new binding is registered;
    - a specification that is no longer learnable loses its binding.

    A learnable specification without a tensor form is rejected (TypeError)
    before anything is written.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    spec : ParameterSpec
        Fully defined parameter specification to store.
    tier : str
        Database tier to write to (default 'calibrated').
    """
    if not isinstance(spec, ParameterSpec):
        raise TypeError("spec must be an instance of ParameterSpec")

    if spec.requires_grad:
        spec.to_tensor(manager.device)
    previous = manager.parameters.get(spec.key)
    if previous is None:
        previous = manager.database.get_parameter(spec.key)
    old_value = previous.value if previous is not None else None

    manager.database.set_parameter(spec, tier=tier)

    manager.parameters[spec.key] = spec
    manager.storage.set(spec.key, spec.value)
    bind_learnable_value(manager, spec)

    if manager._cache_enabled:
        if manager._parameter_cache is not None:
            manager._parameter_cache.clear()
        if manager._unit_conversion_cache is not None:
            manager._unit_conversion_cache.clear()

    if values_differ(old_value, spec.value):
        manager._notify_change(spec.key, old_value, spec.value)


def register_parameter(manager: Any, spec: ParameterSpec) -> None:
    """
    Register a parameter specification with the manager.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    spec : ParameterSpec
        Parameter specification to register.
    """
    existing_spec = manager.parameters.get(spec.key)
    old_value = existing_spec.value if existing_spec is not None else None
    if existing_spec is not None:
        spec_changed = (
            values_differ(existing_spec.value, spec.value)
            or existing_spec.units != spec.units
            or existing_spec.dtype != spec.dtype
            or existing_spec.bounds != spec.bounds
            or existing_spec.category != spec.category
            or existing_spec.description != spec.description
        )
        if spec_changed:
            logger.warning("Overwriting existing parameter: %s", spec.key)
        else:
            logger.debug("Refreshing existing parameter without spec change: %s", spec.key)
        manager._unit_validation_checked.discard(spec.key)

    if manager.validation_level not in [ValidationLevel.NONE, ValidationLevel.BASIC]:
        errors = spec.validate_value(spec.value)
        severe_errors = [
            e for e in errors if "nan" in e.lower() or "inf" in e.lower() or "type" in e.lower()
        ]
        if severe_errors:
            raise ValueError(f"Parameter specification invalid: {severe_errors}")

    if hasattr(manager, "storage") and manager.storage:
        storage_value = manager.storage.get(spec.key)
        if storage_value is not None:
            old_value = storage_value

    manager._check_units_for_spec(spec.key, spec, source="register")

    manager.parameters[spec.key] = spec

    if hasattr(manager, "storage") and manager.storage:
        manager.storage.set(spec.key, spec.value)

    # The live binding follows the set_parameter rule: a re-registration (for
    # example a database reload) keeps an existing binding of the same shape and
    # dtype, its object, gradient flag and device, so the Functors that hold it
    # by reference and the optimizer keep one theta; a changed shape or dtype
    # gets a new binding, and a spec that is not learnable has none.
    bind_learnable_value(manager, spec)

    manager.invalidate_cache_key(spec.key)

    if hasattr(manager, "database"):
        manager.database.cache_dirty = True

    if existing_spec is not None and values_differ(old_value, spec.value):
        manager._notify_change(spec.key, old_value, spec.value)

    logger.debug("Registered parameter: %s", spec.key)


def register_parameters(manager: Any, specs: List[ParameterSpec]) -> None:
    """
    Register multiple parameter specifications.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    specs : List[ParameterSpec]
        Parameter specifications to register.
    """
    for spec in specs:
        register_parameter(manager, spec)


def set_value(
    manager: Any,
    key: str,
    value: ParameterValue,
    units: Optional[str] = None,
    validate: bool = True,
    notify: bool = True,
) -> None:
    """
    Set parameter value with optional validation and notification.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    key : str
        Parameter key (dot notation).
    value : ParameterValue
        New parameter value.
    units : Optional[str]
        Expected units for validation.
    validate : bool
        Whether to validate the new value.
    notify : bool
        Whether to notify change callbacks.
    """
    canonical_key = normalize_parameter_key(key)
    old_value = manager.storage.get(canonical_key)
    normalized_units = normalize_unit_string(units)

    spec = _get_spec_for_set_value(manager, canonical_key)
    # The manager holds its own copy of a written container (the copy rule of
    # caller_owned_value, architecture overview Section 6.1): a caller that
    # changes its container after the write does not change the stored value.
    candidate_value = caller_owned_value(value)

    # References: BIPM SI Brochure (9th ed.), Sect. 2.3.3 (the unit one); RELIC
    #   unit SSOT (data/standards/unit_converter.py).
    # Mathematical expression: v_spec = f(v_caller) for the registered unit pair
    #   (caller unit -> declared unit); f is the identity when both units are
    #   dimensionless aliases ("", "1", "dimensionless").
    # Parameter meanings: v_caller is the value in the caller's unit, v_spec the
    #   value in the specification's declared unit.
    # Units: the declared unit of the specification is kept; it is never
    #   replaced by the caller's unit.
    # Domain/constraints: a pair the unit owner cannot convert raises ValueError
    #   naming the parameter; nothing is written.
    # Sign convention: none.
    if normalized_units is not None and spec.units != normalized_units:
        if not (_is_dimensionless_equivalent(normalized_units) and _is_dimensionless_equivalent(spec.units)):
            try:
                candidate_value = unit_converter.convert(
                    candidate_value,
                    key,
                    normalized_units,
                    spec.units,
                    ConversionContext(),
                )
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                raise ValueError(f"Unit conversion failed for {key}: {exc}") from exc
            logger.debug("Converted %s from %s to %s", key, normalized_units, spec.units)

    final_value = candidate_value
    effective_policy = manager._bounds_policy
    clamped = False
    clamped_value: Optional[float] = None

    if validate:
        final_value, clamped, clamped_value = _validate_and_resolve_value(
            manager=manager,
            key=key,
            canonical_key=canonical_key,
            spec=spec,
            initial_value=final_value,
            effective_policy=effective_policy,
        )

    final_value, clamped, clamped_value = _apply_post_validation_clamp(
        manager=manager,
        spec=spec,
        final_value=final_value,
        validate=validate,
        effective_policy=effective_policy,
        clamped=clamped,
        clamped_value=clamped_value,
    )

    _raise_if_clamped_with_strict_bounds_policy(
        manager=manager,
        canonical_key=canonical_key,
        original_value=value,
        spec=spec,
        validate=validate,
        effective_policy=effective_policy,
        clamped=clamped,
    )

    spec.value = final_value
    manager.parameters[canonical_key] = spec

    manager.storage.set(canonical_key, final_value)

    values_changed = values_differ(old_value, final_value)

    if canonical_key in manager.nn_parameters:
        param_tensor = manager.nn_parameters[canonical_key]
        with torch.no_grad():
            if isinstance(final_value, Tensor):
                param_tensor.data.copy_(final_value.data)
            else:
                param_tensor.data.fill_(float(final_value))

    if values_changed:
        manager.invalidate_cache_key(canonical_key)
    if notify and values_changed:
        manager._notify_change(canonical_key, old_value, final_value)
    if clamped and clamped_value is not None and notify:
        warnings.warn(
            f"Parameter '{canonical_key}' was clamped to {clamped_value} within bounds {getattr(spec, 'bounds', None)}",
            UserWarning,
            stacklevel=3,
        )

    logger.debug("Set parameter %s = %s", canonical_key, final_value)
