"""Validation rules for physical state-domain constraints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from ....data.standards.dimensionality import dimensionality_from_unit
from ..decorators import validated_rule
from ..models import (
    ValidationCode,
    ValidationContext,
    ValidationDetail,
    ValidationLevel,
    ValidationResultModel,
    ValidationResults,
)

_ALLOWED_CONSTRAINT_FIELDS = {
    "bounds",
    "domain",
    "nonnegative",
    "positive",
    "units",
}


def _contains_bool_or_complex(value: Any) -> bool:
    if isinstance(value, Tensor):
        return value.dtype == torch.bool or torch.is_complex(value)
    if isinstance(value, (bool, complex)):
        return True
    dtype_kind = getattr(getattr(value, "dtype", None), "kind", None)
    if dtype_kind in {"b", "c"}:
        return True
    if dtype_kind == "O":
        return any(_contains_bool_or_complex(item) for item in value.flat)
    if isinstance(value, Mapping):
        return any(_contains_bool_or_complex(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_bool_or_complex(item) for item in value)
    return False


def _as_tensor(value: Any) -> Tensor | None:
    if _contains_bool_or_complex(value):
        return None
    if isinstance(value, Tensor) and (
        value.device.type == "meta"
        or value.layout != torch.strided
        or value.is_quantized
    ):
        return None
    try:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError):
        return None
    if (
        tensor.device.type == "meta"
        or tensor.layout != torch.strided
        or tensor.is_quantized
        or tensor.dtype == torch.bool
        or torch.is_complex(tensor)
    ):
        return None
    return tensor


def _failure(variable: str, message: str) -> ValidationResultModel:
    return ValidationResultModel(
        level=ValidationLevel.ERROR,
        code=ValidationCode.GENERIC_VALIDATION_FAILED,
        message=message,
        passed=False,
        layer="L1",
        scope="state_domain_structure",
        tags=["state_domain", "structural", "physical_domain"],
        details=[ValidationDetail(variable_name=variable, value=message)],
    )


def _bound_tensor(value: Any, *, device: torch.device) -> Tensor | None:
    """Return one finite real scalar bound on the state device."""

    if _contains_bool_or_complex(value):
        return None
    if isinstance(value, Tensor) and (
        value.device.type == "meta"
        or value.layout != torch.strided
        or value.is_quantized
    ):
        return None
    try:
        bound = torch.as_tensor(value, device=device)
    except (TypeError, ValueError, RuntimeError):
        return None
    if (
        bound.device.type == "meta"
        or bound.layout != torch.strided
        or bound.is_quantized
        or bound.numel() != 1
        or bound.is_complex()
        or bound.dtype == torch.bool
    ):
        return None
    if not bool(torch.isfinite(bound).all()):
        return None
    return bound


def validate_state_domain_payload(
    state: Any,
    constraints: Any = None,
    *,
    strict: bool = False,
    required_state_vars: Sequence[str] | None = None,
) -> ValidationResultModel:
    """Validate declared real state values and physical bounds.

    Mathematical expression:
      x in R, x >= 0, x > 0, or lower <= x <= upper as declared.
    Evidence boundary:
      The payload is structural assertion data, not a live StateManager snapshot.
      Full prognostic coverage, runtime shape/dtype, support, and ownership remain [N].
    """

    if not isinstance(strict, bool):
        return _failure("strict", "strict must be bool.")
    if not isinstance(state, Mapping):
        return _failure("state", "state must be a Mapping.")
    if constraints is None:
        constraints = {}
    elif not isinstance(constraints, Mapping):
        return _failure("constraints", "constraints must be a Mapping.")
    for mapping_name, mapping in (("state", state), ("constraints", constraints)):
        if any(
            not isinstance(key, str)
            or not key
            or key != key.strip()
            for key in mapping
        ):
            return _failure(
                mapping_name,
                f"{mapping_name} identities must be exact non-empty strings.",
            )
    required_names: tuple[str, ...] = ()
    if required_state_vars is not None:
        if isinstance(required_state_vars, (str, bytes)) or not isinstance(
            required_state_vars, Sequence
        ):
            return _failure(
                "required_state_vars",
                "required_state_vars must be a sequence of exact non-empty strings.",
            )
        required_names = tuple(required_state_vars)
        if strict and not required_names:
            return _failure(
                "required_state_vars",
                "Strict state-domain validation requires non-empty required_state_vars.",
            )
        if any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in required_names
        ):
            return _failure(
                "required_state_vars",
                "required_state_vars must contain exact non-empty strings.",
            )
        if len(set(required_names)) != len(required_names):
            return _failure(
                "required_state_vars",
                "required_state_vars identities must be unique.",
            )
        missing_required = sorted(set(required_names).difference(state))
        if missing_required:
            return _failure(
                missing_required[0],
                "State-domain evidence is missing graph-required state variables: "
                + ",".join(missing_required),
            )
    if strict and not state:
        return _failure("state", "Strict state-domain validation requires non-empty state.")
    if strict:
        missing_constraints = sorted(set(state).difference(constraints))
        if missing_constraints:
            return _failure(
                missing_constraints[0],
                "Strict state-domain validation requires constraint coverage for: "
                + ",".join(missing_constraints),
            )
        extra_constraints = sorted(set(constraints).difference(state))
        if extra_constraints:
            return _failure(
                extra_constraints[0],
                "State-domain constraints reference absent state variables: "
                + ",".join(extra_constraints),
            )
    checked = 0
    covered = 0
    for variable, value in state.items():
        tensor = _as_tensor(value)
        if tensor is None:
            return _failure(
                variable,
                f"State variable '{variable}' must be a real-valued dense tensor.",
            )
        if strict and tensor.numel() == 0:
            return _failure(
                variable,
                f"State variable '{variable}' must be non-empty.",
            )
        if not bool(torch.isfinite(tensor).all().item()):
            return _failure(variable, f"State variable '{variable}' must be finite.")
        checked += 1
        rule = constraints.get(variable, {})
        if not isinstance(rule, Mapping):
            if strict:
                return _failure(
                    variable,
                    f"State variable '{variable}' requires a mapping domain contract.",
                )
            continue
        for flag_name in ("nonnegative", "positive"):
            if flag_name in rule and not isinstance(rule.get(flag_name), bool):
                return _failure(
                    variable,
                    f"State variable '{variable}' {flag_name} must be bool.",
                )
        if strict:
            invalid_fields = sorted(
                key
                for key in rule
                if not isinstance(key, str) or key not in _ALLOWED_CONSTRAINT_FIELDS
            )
            if invalid_fields:
                return _failure(
                    variable,
                    f"State variable '{variable}' requires canonical constraint fields; "
                    f"invalid={invalid_fields}.",
                )
            units = rule.get("units")
            if not isinstance(units, str) or not units.strip():
                return _failure(
                    variable,
                    f"State variable '{variable}' requires non-empty units.",
                )
            if dimensionality_from_unit(units) is None:
                return _failure(
                    variable,
                    f"State variable '{variable}' requires parseable units.",
                )
            raw_domain = rule.get("domain", "")
            if "domain" in rule and not isinstance(raw_domain, str):
                return _failure(
                    variable,
                    f"State variable '{variable}' domain must be a string.",
                )
            explicit_domain = raw_domain.strip().lower() if isinstance(raw_domain, str) else ""
            has_bounds = "bounds" in rule
            if not (
                rule.get("nonnegative") is True
                or rule.get("positive") is True
                or has_bounds
                or explicit_domain == "real"
            ):
                return _failure(
                    variable,
                    f"State variable '{variable}' requires an explicit physical domain.",
                )
            if explicit_domain and explicit_domain != "real":
                return _failure(
                    variable,
                    f"State variable '{variable}' has an unrecognized domain.",
                )
            covered += 1
        lower_tensor: Tensor | None = None
        upper_tensor: Tensor | None = None
        if "bounds" in rule:
            bounds = rule.get("bounds")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                return _failure(
                    variable,
                    f"State variable '{variable}' bounds must contain lower and upper values.",
                )
            lower, upper = bounds
            if lower is None and upper is None:
                return _failure(
                    variable,
                    f"State variable '{variable}' bounds require at least one endpoint.",
                )
            lower_tensor = (
                _bound_tensor(lower, device=tensor.device) if lower is not None else None
            )
            upper_tensor = (
                _bound_tensor(upper, device=tensor.device) if upper is not None else None
            )
            if (lower is not None and lower_tensor is None) or (
                upper is not None and upper_tensor is None
            ):
                return _failure(
                    variable,
                    f"State variable '{variable}' bounds must be finite real numeric scalars.",
                )
            if (
                lower_tensor is not None
                and upper_tensor is not None
                and bool((lower_tensor > upper_tensor).any())
            ):
                return _failure(
                    variable,
                    f"State variable '{variable}' bounds must be ordered lower <= upper.",
                )
            if (
                rule.get("positive") is True
                and upper_tensor is not None
                and bool((upper_tensor <= 0).item())
            ):
                return _failure(
                    variable,
                    f"State variable '{variable}' positive constraint conflicts with upper bound.",
                )
            if (
                rule.get("nonnegative") is True
                and upper_tensor is not None
                and bool((upper_tensor < 0).item())
            ):
                return _failure(
                    variable,
                    f"State variable '{variable}' nonnegative constraint conflicts with upper bound.",
                )
            if lower_tensor is not None and bool((tensor < lower_tensor).any()):
                return _failure(variable, f"State variable '{variable}' is below its lower bound.")
            if upper_tensor is not None and bool((tensor > upper_tensor).any()):
                return _failure(variable, f"State variable '{variable}' is above its upper bound.")
        if rule.get("nonnegative") is True and bool((tensor < 0).any()):
            return _failure(variable, f"State variable '{variable}' must be nonnegative.")
        if rule.get("positive") is True and bool((tensor <= 0).any()):
            return _failure(variable, f"State variable '{variable}' must be positive.")

    return ValidationResultModel(
        level=ValidationLevel.INFO,
        code=ValidationCode.GENERIC_VALIDATION_OK,
        message="State-domain constraints passed.",
        passed=True,
        layer="L1",
        scope="state_domain_structure",
        tags=["state_domain", "structural", "physical_domain"],
        metrics={
            "checked_count": checked,
            "covered_count": covered,
            "required_count": len(required_names),
            "extra_count": (
                len(set(state).difference(required_names))
                if required_names
                else 0
            ),
        },
    )


@validated_rule(
    "state_domain_structure",
    "L1",
    name="State domain structure",
    description="Validate finite state values and declared physical-domain bounds.",
    domains=["computation"],
    tags=["state_domain", "structural", "physical_domain"],
    references=[
        "RELIC-IPHS-DEC Typed Cochain Morphism Enhancement Plan",
        "biogeochemical systems engineering principles: transaction checks",
    ],
)
def validate_state_domain_structure(context: ValidationContext) -> ValidationResults:
    """Validate state-domain constraints from ValidationContext."""

    state = context.state
    metadata = context.metadata if isinstance(context.metadata, dict) else {}
    constraints = metadata.get("state_domain_constraints", {})
    return [
        validate_state_domain_payload(
            state,
            constraints,
            strict=metadata.get("strict") is True,
            required_state_vars=metadata.get("required_state_vars"),
        )
    ]


__all__ = ["validate_state_domain_payload", "validate_state_domain_structure"]
