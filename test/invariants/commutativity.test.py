"""I5: Order independence for commutative increment pairs.

References:
- Abelian group property of addition

Mathematical expression:
S + delta_a + delta_b = S + delta_b + delta_a

Parameters:
S state value (kg), delta_a increment (kg per step), delta_b increment (kg per step)
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from relic.core.state import StateManager, UpdateItem, StateVariableSpec
from relic.core.state import UpdateMode


class TestCommutativity:
    """Test commutativity of independent increments."""

    def test_independent_increments_commute(self) -> None:
        """Applying two independent increments in either order yields same state."""
        inc_a = UpdateItem(value=3.0, unit="kg", basis="per_step")
        inc_b = UpdateItem(value=7.0, unit="kg", basis="per_step")

        sm_ab = self._build_manager()
        sm_ab.apply_increment("scalar_state", inc_a)
        sm_ab.apply_increment("scalar_state", inc_b)
        val_ab = sm_ab.get_state()["scalar_state"].item()

        sm_ba = self._build_manager()
        sm_ba.apply_increment("scalar_state", inc_b)
        sm_ba.apply_increment("scalar_state", inc_a)
        val_ba = sm_ba.get_state()["scalar_state"].item()

        assert pytest.approx(val_ab, rel=1e-9) == val_ba

    def test_multi_variable_increments_commute(self) -> None:
        """Increments to different variables are commutative."""
        sm_ab = self._build_manager_dual()
        sm_ba = self._build_manager_dual()
        inc_carbon = UpdateItem(value=10.0, unit="kg C", basis="per_step")
        inc_nitrogen = UpdateItem(value=2.0, unit="kg N", basis="per_step")

        sm_ab.apply_updates({"carbon_pool": inc_carbon}, check_conservation=False)
        sm_ab.apply_updates({"nitrogen_pool": inc_nitrogen}, check_conservation=False)

        sm_ba.apply_updates({"nitrogen_pool": inc_nitrogen}, check_conservation=False)
        sm_ba.apply_updates({"carbon_pool": inc_carbon}, check_conservation=False)

        state_ab = sm_ab.get_state()
        state_ba = sm_ba.get_state()

        assert pytest.approx(state_ab["carbon_pool"].item(), rel=1e-9) == state_ba["carbon_pool"].item()
        assert pytest.approx(state_ab["nitrogen_pool"].item(), rel=1e-9) == state_ba["nitrogen_pool"].item()

    @staticmethod
    def _build_manager() -> StateManager:
        sm = StateManager()
        sm.register_variable(
            StateVariableSpec(
                name="scalar_state",
                units="kg",
                conserved=False,
                default_value=0.0,
            )
        )
        return sm

    @staticmethod
    def _build_manager_dual() -> StateManager:
        sm = StateManager()
        sm.register_variable(
            StateVariableSpec(
                name="carbon_pool",
                units="kg C",
                conserved=True,
                default_value=100.0,
            )
        )
        sm.register_variable(
            StateVariableSpec(
                name="nitrogen_pool",
                units="kg N",
                conserved=True,
                default_value=20.0,
            )
        )
        return sm


@pytest.mark.property
class TestCommutativityPBT:
    """I5 property-based commutativity for independent ACCUMULATE updates."""

    @given(
        initial=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        delta_a=st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
        delta_b=st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_accumulate_order_independent(
        self,
        initial: float,
        delta_a: float,
        delta_b: float,
    ) -> None:
        """apply(a then b) equals apply(b then a)."""
        def _manager() -> StateManager:
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

        sm_ab = _manager()
        sm_ba = _manager()

        item_a = UpdateItem(value=delta_a, mode=UpdateMode.ACCUMULATE, unit="kg", basis="per_step")
        item_b = UpdateItem(value=delta_b, mode=UpdateMode.ACCUMULATE, unit="kg", basis="per_step")

        sm_ab.apply_increment("x", item_a, dt=1.0)
        sm_ab.apply_increment("x", item_b, dt=1.0)

        sm_ba.apply_increment("x", item_b, dt=1.0)
        sm_ba.apply_increment("x", item_a, dt=1.0)

        val_ab = float(sm_ab.get_state()["x"].item())
        val_ba = float(sm_ba.get_state()["x"].item())
        tol = max(1e-10, 1e-10 * abs(val_ab))
        assert abs(val_ab - val_ba) <= tol
