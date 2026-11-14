"""I4 reversibility invariant tests with PBT.

References:
- Additive group inverse property for increment operators.
- RELIC StateManager ACCUMULATE semantics.

Mathematical expression:
- apply(state, +delta) then apply(state, -delta) returns original state.

Parameter meanings:
- initial: initial scalar state value.
- delta: per-step increment magnitude.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


@pytest.mark.property
class TestReversibilityPBT:
    """I4 property tests for additive inverse behavior."""

    @given(
        initial=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        delta=st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_add_then_subtract_restores_state(self, initial: float, delta: float) -> None:
        """Applying +delta then -delta restores initial scalar state."""
        sm = StateManager()
        sm.register_variable(
            StateVariableSpec(
                name="x",
                units="kg",
                conserved=False,
                default_value=initial,
            )
        )
        before = float(sm.get_state()["x"].item())

        sm.apply_increment(
            "x",
            UpdateItem(value=delta, mode=UpdateMode.ACCUMULATE, unit="kg", basis="per_step"),
            dt=1.0,
        )
        sm.apply_increment(
            "x",
            UpdateItem(value=-delta, mode=UpdateMode.ACCUMULATE, unit="kg", basis="per_step"),
            dt=1.0,
        )

        after = float(sm.get_state()["x"].item())
        tol = max(1e-10, 1e-10 * abs(before))
        assert abs(after - before) <= tol
