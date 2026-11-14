"""I1 no-side-effects invariant tests with PBT.

References:
- RELIC architectural contract: forward-like read operations must be pure.
- Functional purity principle for state-read paths.

Mathematical expression:
- Let S0 be a state snapshot. For read-only operation R:
  R(S0) does not change state, so snapshot(R(S0)) = S0.

Parameter meanings:
- initial: initial scalar state.
- probe: arbitrary probe value used to construct temporary UpdateItem objects.
"""

import copy

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from relic.core.state import StateManager, StateVariableSpec, UpdateItem, UpdateMode


def _snapshot(sm: StateManager) -> dict:
    """Create a deep snapshot of StateManager state."""
    captured = {}
    for key, value in sm.get_state().items():
        if torch.is_tensor(value):
            captured[key] = value.clone()
        else:
            captured[key] = copy.deepcopy(value)
    return captured


@pytest.mark.property
class TestNoSideEffectsPBT:
    """I1 property tests for non-mutating read-style paths."""

    @given(
        initial=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        probe=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_read_paths_do_not_mutate_state(self, initial: float, probe: float) -> None:
        """Constructing updates and reading state should be side-effect free."""
        sm = StateManager()
        sm.register_variable(
            StateVariableSpec(
                name="x",
                units="kg",
                conserved=False,
                default_value=initial,
            )
        )

        before = _snapshot(sm)

        _ = sm.get_state()
        _ = sm.get_state().get("x")
        _ = UpdateItem(value=probe, mode=UpdateMode.ACCUMULATE, unit="kg", basis="per_step")

        after = _snapshot(sm)
        for key in before:
            if torch.is_tensor(before[key]):
                assert torch.equal(before[key], after[key])
            else:
                assert before[key] == after[key]
