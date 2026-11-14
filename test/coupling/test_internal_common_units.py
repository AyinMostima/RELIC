import types

import pytest

from relic.coupling.internal import common


class _StubCoupler(common.CouplingParameterMixin):
    def __init__(self, error):
        self._error = error

    def get_parameter_value(self, key: str, **_kwargs):
        raise self._error


@pytest.mark.parametrize("required", [True, False])
def test_coupling_parameter_errors_propagate(required):
    """A failing lookup is an error for a required and for an optional parameter; the
    mixin no longer turns it into None (an absent optional key is None from the manager
    itself, without an exception)."""
    coupler = _StubCoupler(ValueError("units of coupling.test.foo differ"))
    with pytest.raises(ValueError, match="units"):
        coupler._get_coupling_parameter(
            "foo", namespace="coupling.test", units="K", required=required
        )


def test_dead_common_helpers_are_removed():
    """apply_state_defaults, _load_param_specs and its environment switch had no caller."""
    assert not hasattr(common, "apply_state_defaults")
    assert not hasattr(common, "STRICT_UNIT_ENFORCEMENT")
    assert not hasattr(common.CouplingParameterMixin, "_load_param_specs")
