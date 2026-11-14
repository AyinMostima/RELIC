"""Shared fixtures for invariants tests."""

import copy

import pytest
import torch
from hypothesis import strategies as st

from relic.core.state import StateManager, StateVariableSpec

# Shared Hypothesis strategies for invariant PBT/MT.
finite_floats = st.floats(
    min_value=-1e12,
    max_value=1e12,
    allow_nan=False,
    allow_infinity=False,
)

positive_floats = st.floats(
    min_value=1e-8,
    max_value=1e8,
    allow_nan=False,
    allow_infinity=False,
)

scaling_factors = st.floats(
    min_value=0.01,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)

basis_strategy = st.sampled_from(["per_second", "per_step"])


@pytest.fixture
def state_manager() -> StateManager:
    sm = StateManager()
    sm.register_variable(
        StateVariableSpec(
            name="scalar_state",
            units="kg",
            conserved=False,
            default_value=0.0,
        )
    )
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
            name="carbon_loss",
            units="kg C",
            conserved=True,
            default_value=0.0,
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


@pytest.fixture
def bounded_state_manager() -> StateManager:
    """StateManager with bounded variables for I7 testing."""
    spec = StateVariableSpec(
        name="bounded_pool",
        units="kg",
        bounds=(0.0, 100.0),
        conserved=True,
    )
    sm = StateManager(schema={"bounded_pool": spec})
    sm.set_state({"bounded_pool": torch.tensor(50.0)})
    return sm


@pytest.fixture
def snapshot():
    def _snapshot(sm: StateManager):
        return {
            key: (value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
            for key, value in sm.state_vars.items()
        }

    return _snapshot
