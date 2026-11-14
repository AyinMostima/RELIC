import dataclasses
import re

import pytest
import torch

from relic.coupling.core.policy import get_precision_policy, ScalingMode, _POLICY_CACHE
from relic.parameters import get_global_manager


@pytest.fixture(autouse=True)
def reset_policy_cache():
    _POLICY_CACHE.clear()
    yield
    _POLICY_CACHE.clear()


def test_strict_conservative_policy_loaded_from_params():
    # Ensure ParameterManager is initialized so parameter-backed policies are used
    get_global_manager()
    precision_policy = get_precision_policy(ScalingMode.STRICT_CONSERVATIVE)
    assert abs(precision_policy.tolerance - 1.0e-12) < 1e-15
    assert precision_policy.relative_tolerance == precision_policy.tolerance
    assert abs(precision_policy.absolute_tolerance - 1.0e-14) < 1e-16
    assert precision_policy.dtype == torch.float64
    assert precision_policy.enforce_conservation is True
    assert precision_policy.max_iterations == 1000


def test_balanced_policy_loaded_from_params():
    get_global_manager()
    precision_policy = get_precision_policy(ScalingMode.BALANCED)
    assert abs(precision_policy.tolerance - 1.0e-6) < 1e-12
    assert precision_policy.dtype == torch.float32
    assert precision_policy.enforce_conservation is False
    assert precision_policy.max_iterations == 100


def test_policy_respects_parameter_manager_overrides():
    pm = get_global_manager()
    ns = "coupling.scales.defaults.precision.fast_approx"
    key = f"{ns}.tolerance"
    original = pm.get("tolerance", namespace=ns, expected_type=float)

    try:
        pm.storage.set(key, 2.5e-4)
        pm.clear_cache()
        precision_policy = get_precision_policy(ScalingMode.FAST_APPROX, parameter_manager=pm)
        assert precision_policy.tolerance == pytest.approx(2.5e-4)
    finally:
        pm.storage.set(key, original)
        pm.clear_cache()


def test_non_finite_tolerance_is_an_error_naming_its_key():
    """A NaN tolerance passes the bounds check of the store; the policy rejects it by name."""
    pm = get_global_manager()
    ns = "coupling.scales.defaults.precision.conservative"
    key = f"{ns}.tolerance"
    original = pm.get("tolerance", namespace=ns, expected_type=float)

    try:
        pm.storage.set(key, float("nan"))
        pm.clear_cache()
        with pytest.raises(RuntimeError, match=re.escape(key)):
            get_precision_policy(ScalingMode.CONSERVATIVE, parameter_manager=pm)
    finally:
        pm.storage.set(key, original)
        pm.clear_cache()


def test_policy_has_one_tolerance_rule():
    """Every PrecisionPolicy value comes from the parameter store; no derived or per-variable rule."""
    import relic.coupling.core.policy as policy

    for name in (
        "update_policy",
        "get_tolerance_for_variable",
        "should_enforce_conservation",
        "get_dtype_for_mode",
        "validate_mode",
    ):
        assert not hasattr(policy, name), name
    for field in dataclasses.fields(policy.PrecisionPolicy):
        assert field.default is dataclasses.MISSING, field.name
        assert field.default_factory is dataclasses.MISSING, field.name


def test_precision_policy_requires_parameters():
    class MissingParameterManager:
        def get(self, *_args, **_kwargs):
            raise ValueError("missing parameter")

    with pytest.raises(RuntimeError):
        get_precision_policy(ScalingMode.CONSERVATIVE, parameter_manager=MissingParameterManager())

