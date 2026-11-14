from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from ..spec import ParameterSpec


def _nearest_float32_inside(bound: float, lower: float, upper: float) -> Optional[float]:
    """Return the float32 value nearest to ``bound`` inside [lower, upper], or None."""
    # References:
    #   - IEEE Std 754-2019, Sect. 4.3 (rounding-direction attributes) and
    #     Sect. 5.3.1 (nextUp and nextDown).
    #   - Goldberg (1991), ACM Computing Surveys 23(1), 5-48,
    #     doi:10.1145/103162.103163.
    # Mathematical expression:
    #   q = RN32(bound)
    #   c = nextDown32(q) if q > upper; c = nextUp32(q) if q < lower; else c = q
    #   result = c if lower <= c <= upper, otherwise None
    # Parameter meanings:
    #   bound: violated endpoint of the declared interval (lower or upper)
    #   RN32: rounding to the nearest float32 value (ties to even)
    #   nextDown32, nextUp32: adjacent float32 values below and above
    #   c: clamped value
    # Units:
    #   bound, lower, upper, q, and c are in the declared unit of the
    #   parameter; rounding leaves the unit unchanged.
    # Domain/constraints:
    #   lower <= bound <= upper. q and c are compared with lower and upper as
    #   exact float64 values. q differs from bound by at most half a float32
    #   spacing, so one step reaches the nearest float32 value on the interval
    #   side; when that value is outside as well, the interval holds no float32
    #   value and the result is None. A NaN bound gives None.
    # Sign convention:
    #   none; the step always points from the violated bound into the interval.
    candidate = torch.tensor(bound, dtype=torch.float32)
    if float(candidate) > upper:
        candidate = torch.nextafter(candidate, torch.tensor(-math.inf, dtype=torch.float32))
    elif float(candidate) < lower:
        candidate = torch.nextafter(candidate, torch.tensor(math.inf, dtype=torch.float32))
    inside = float(candidate)
    return inside if lower <= inside <= upper else None


def clamp_value_to_bounds(
    manager: Any,
    spec: ParameterSpec,
    attempted_value: Any,
) -> Tuple[bool, Optional[Any]]:
    """Clamp numeric parameters to configured bounds when possible."""
    bounds = getattr(spec, "bounds", None)
    if not bounds:
        return False, None
    min_val, max_val = bounds
    try:
        numeric_value = float(attempted_value)
    except (TypeError, ValueError):
        return False, None
    # References:
    #   - IEEE Std 754-2019, Sect. 5.11 (every ordered comparison with a NaN
    #     operand is false) and Sect. 6.1 (infinity arithmetic).
    # Mathematical expression:
    #   v_c = min(max(v, lower), upper), defined for finite v only
    # Parameter meanings:
    #   v: requested value; lower, upper: declared bounds (None is an open
    #   side); v_c: clamped value
    # Units:
    #   v, lower, upper and v_c are in the declared unit of the parameter.
    # Domain/constraints:
    #   v must be finite. NaN fails every order comparison and has no nearest
    #   admissible value; an infinite request reports an upstream overflow
    #   rather than a value near a bound. Neither is clamped: (False, None) is
    #   returned, so the caller keeps the bounds error and rejects the write.
    # Sign convention:
    #   none.
    if not math.isfinite(numeric_value):
        return False, None
    clamped_value = numeric_value
    if min_val is not None and numeric_value < min_val:
        clamped_value = float(min_val)
    if max_val is not None and clamped_value > max_val:
        clamped_value = float(max_val)
    if clamped_value == numeric_value:
        return False, None

    dtype_str = (spec.dtype or "").lower()
    original_tensor = isinstance(attempted_value, Tensor)
    target_device = attempted_value.device if original_tensor else None
    target_dtype = attempted_value.dtype if original_tensor else None

    # A float32 request tensor and a scalar request on a parameter declared
    # float32 are stored as the float32 value nearest to the violated bound
    # inside the declared interval; rounding the bound itself to float32 can
    # leave the interval. An interval that holds no float32 value keeps the
    # declared bound.
    if original_tensor:
        stored_as_float32 = target_dtype == torch.float32
    else:
        stored_as_float32 = dtype_str in {"float32", "float"}
    if stored_as_float32:
        inside = _nearest_float32_inside(
            clamped_value,
            -math.inf if min_val is None else float(min_val),
            math.inf if max_val is None else float(max_val),
        )
        if inside is not None:
            clamped_value = inside

    message = (
        f"Parameter '{spec.key}' value {numeric_value} outside bounds "
        f"[{min_val}, {max_val}]; clamped to {clamped_value}"
    )
    warnings.warn(message, UserWarning, stacklevel=4)
    manager._record_clamp_event(spec, attempted_value, clamped_value, message=message)

    if original_tensor:
        tensor_value = torch.as_tensor(clamped_value, device=target_device, dtype=target_dtype)
        return True, tensor_value
    return True, clamped_value
