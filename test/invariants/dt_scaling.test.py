"""I6 dt-scaling invariant tests with PBT/MT.

References:
- Chen, T. Y., et al. (2018). Metamorphic Testing: A Review of Challenges and Opportunities.
- RELIC state contract: UpdateItem.basis controls temporal integration semantics.

Mathematical expressions:
- MR-A1 (per_second linearity): Delta(x; s*dt) = s * Delta(x; dt)
- MR-A2 (per_step invariance): Delta(x; dt1) = Delta(x; dt2)
- MR-A3 (split equivalence): Delta(x; dt_a + dt_b) = Delta(x; dt_a) + Delta(x; dt_b)

Parameter meanings:
- flux: process rate in per-second basis.
- dt, dt_a, dt_b: time-step length in seconds.
- s: dimensionless scaling factor.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


def _fresh_state_manager(initial: float = 0.0) -> StateManager:
    """Build a fresh scalar state manager."""
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="x",
            units="kg",
            conserved=False,
            default_value=initial,
        )
    )
    return sm


def _state_value(sm: StateManager, key: str = "x") -> float:
    """Return scalar state value as float."""
    return float(sm.get_state()[key].item())


@pytest.mark.property
class TestDtScalingMRA:
    """MR-A suite for dt and UpdateItem basis semantics."""

    @given(
        flux=st.floats(min_value=1e-8, max_value=1e6, allow_nan=False, allow_infinity=False),
        dt=st.floats(min_value=1e-8, max_value=1e5, allow_nan=False, allow_infinity=False),
        scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_per_second_linear_scaling(self, flux: float, dt: float, scale: float) -> None:
        """MR-A1: per_second increment scales linearly with dt."""
        inc = UpdateItem(value=flux, mode=UpdateMode.ACCUMULATE, basis="per_second")
        sm_ref = _fresh_state_manager()
        sm_scaled = _fresh_state_manager()

        sm_ref.apply_increment("x", inc, dt=dt)
        sm_scaled.apply_increment("x", inc, dt=dt * scale)

        delta_ref = _state_value(sm_ref)
        delta_scaled = _state_value(sm_scaled)
        expected = delta_ref * scale
        tol = max(1e-8, 1e-8 * abs(expected))
        assert abs(delta_scaled - expected) <= tol

    @given(
        value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        dt1=st.floats(min_value=1e-8, max_value=1e6, allow_nan=False, allow_infinity=False),
        dt2=st.floats(min_value=1e-8, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_per_step_dt_invariance(self, value: float, dt1: float, dt2: float) -> None:
        """MR-A2: per_step increment is invariant to dt."""
        inc = UpdateItem(value=value, mode=UpdateMode.ACCUMULATE, basis="per_step")
        sm_1 = _fresh_state_manager()
        sm_2 = _fresh_state_manager()

        sm_1.apply_increment("x", inc, dt=dt1)
        sm_2.apply_increment("x", inc, dt=dt2)

        v1 = _state_value(sm_1)
        v2 = _state_value(sm_2)
        tol = max(1e-10, 1e-10 * abs(v1))
        assert abs(v1 - v2) <= tol

    @given(
        flux=st.floats(min_value=1e-8, max_value=1e6, allow_nan=False, allow_infinity=False),
        dt_a=st.floats(min_value=1e-8, max_value=1e5, allow_nan=False, allow_infinity=False),
        dt_b=st.floats(min_value=1e-8, max_value=1e5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_split_equivalence(self, flux: float, dt_a: float, dt_b: float) -> None:
        """MR-A3: single large step equals sum of split steps for per_second."""
        inc = UpdateItem(value=flux, mode=UpdateMode.ACCUMULATE, basis="per_second")
        sm_combined = _fresh_state_manager()
        sm_split = _fresh_state_manager()

        sm_combined.apply_increment("x", inc, dt=dt_a + dt_b)
        sm_split.apply_increment("x", inc, dt=dt_a)
        sm_split.apply_increment("x", inc, dt=dt_b)

        v_combined = _state_value(sm_combined)
        v_split = _state_value(sm_split)
        tol = max(1e-8, 1e-8 * abs(v_combined))
        assert abs(v_combined - v_split) <= tol
