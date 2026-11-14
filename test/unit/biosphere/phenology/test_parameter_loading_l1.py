"""L1 tests for phenology parameter primitives under unified access rules."""

import pytest

from relic.parameters import get_global_manager


def test_global_parameter_manager_available() -> None:
    pm = get_global_manager()
    assert pm is not None
