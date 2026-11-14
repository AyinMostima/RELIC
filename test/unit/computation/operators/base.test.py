"""Direct L1 tests for the canonical numerical operator lifecycle."""

from __future__ import annotations

import pytest
import torch

from relic.computation.operators.gradient import CentralDifferenceGradient1D


def _gradient_operator(
    *,
    validate_parameters: bool = True,
    **config: object,
) -> CentralDifferenceGradient1D:
    resolved = {
        "dx": 1.0,
        "boundary_method": "extrapolate",
        "order": 2,
        **config,
    }
    return CentralDifferenceGradient1D(
        name="base_lifecycle_probe",
        config=resolved,
        dtype=torch.float32,
        validate_parameters=validate_parameters,
    )


def _operator_snapshot(operator):
    tensors = {key: value.clone() for key, value in operator.state_dict().items()}
    attributes = {
        key: value for key, value in vars(operator).items()
        if not key.startswith("_") and not isinstance(value, torch.Tensor)
    }
    return tensors, attributes


def test_forward_and_direct_apply_change_no_operator_state() -> None:
    """I1: an operator call changes neither its tensors nor its attributes."""
    operator = _gradient_operator()
    field = torch.arange(5, dtype=torch.float32).reshape(1, -1)
    tensors, attributes = _operator_snapshot(operator)

    operator.apply(field)
    operator(field)

    after_tensors, after_attributes = _operator_snapshot(operator)
    assert after_tensors.keys() == tensors.keys()
    for key, value in after_tensors.items():
        assert torch.equal(value, tensors[key]), key
    assert after_attributes == attributes
    assert not hasattr(operator, "apply_count")
    assert not hasattr(operator, "reset_diagnostics")
    assert "apply_count" not in operator.get_diagnostics()


def test_to_supports_standard_module_dtype_keyword() -> None:
    operator = _gradient_operator()

    returned = operator.to(dtype=torch.float64)
    result = operator(torch.arange(5, dtype=torch.float32).reshape(1, -1))

    assert returned is operator
    assert operator.dtype == torch.float64
    assert result.dtype == torch.float64
    assert operator.device == result.device


def test_config_parameter_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="outside bounds"):
        _gradient_operator(dx=-1.0)


def test_config_parameter_units_are_validated() -> None:
    with pytest.raises(ValueError, match="units mismatch"):
        _gradient_operator(dx={"value": 1.0, "units": "s"})


# ---------------------------------------------------------------------------
# Strict parameter resolution: no Python defaults, nothing swallowed.
# ---------------------------------------------------------------------------


class _OverridingManager:
    """Delegate to the real manager except for the listed keys."""

    def __init__(self, inner, overrides):
        self._inner = inner
        self._overrides = overrides

    def get(self, key, *args, **kwargs):
        if key in self._overrides:
            override = self._overrides[key]
            if isinstance(override, Exception):
                raise override
            return override
        return self._inner.get(key, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _patch_manager(monkeypatch, overrides) -> None:
    import relic.computation.operators.base as base_module

    real_get_global_manager = base_module.get_global_manager
    monkeypatch.setattr(
        base_module,
        "get_global_manager",
        lambda *args, **kwargs: _OverridingManager(
            real_get_global_manager(*args, **kwargs), overrides
        ),
    )


def test_get_parameter_has_no_python_default() -> None:
    import inspect

    from relic.computation.operators.base import OperatorBase

    assert "default" not in inspect.signature(OperatorBase.get_parameter).parameters


def test_manager_bootstrap_failure_propagates(monkeypatch) -> None:
    import relic.computation.operators.base as base_module

    def unavailable_manager(*args, **kwargs):
        raise RuntimeError("parameter manager unavailable")

    monkeypatch.setattr(base_module, "get_global_manager", unavailable_manager)
    with pytest.raises(RuntimeError, match="parameter manager unavailable"):
        _gradient_operator()


def test_manager_lookup_errors_propagate(monkeypatch) -> None:
    _patch_manager(
        monkeypatch,
        {"computation.operators.gradient.order": ValueError("invalid gradient order")},
    )
    with pytest.raises(ValueError, match="invalid gradient order"):
        CentralDifferenceGradient1D(
            name="lookup_error_probe", config={"dx": 1.0, "boundary_method": "zero"}
        )


def test_out_of_bounds_manager_value_surfaces_as_out_of_bounds(monkeypatch) -> None:
    _patch_manager(monkeypatch, {"computation.operators.gradient.dx": -1.0})
    with pytest.raises(ValueError, match="outside bounds"):
        CentralDifferenceGradient1D(
            name="bounds_probe", config={"boundary_method": "zero", "order": 2}
        )


def test_get_parameter_takes_bounds_only_from_yaml() -> None:
    import inspect

    from relic.computation.operators.base import OperatorBase

    assert "bounds" not in inspect.signature(OperatorBase.get_parameter).parameters


def test_get_parameter_takes_units_only_from_yaml() -> None:
    import inspect

    from relic.computation.operators.base import OperatorBase

    assert "units" not in inspect.signature(OperatorBase.get_parameter).parameters


def test_config_override_is_interpreted_in_the_yaml_unit() -> None:
    """ParameterManager contract: convertible units are converted, others rejected."""
    assert _gradient_operator(dx=2.0).dx == 2.0
    assert _gradient_operator(dx={"value": 100.0, "units": "cm"}).dx == pytest.approx(1.0)
    with pytest.raises(ValueError, match="units mismatch"):
        _gradient_operator(dx={"value": 1.0, "units": "s"})


def test_yaml_units_not_code_units_define_the_target_unit(monkeypatch) -> None:
    from types import SimpleNamespace

    import relic.computation.operators.base as base_module

    real_get_global_manager = base_module.get_global_manager

    class _KilometreManager:
        def __init__(self, inner):
            self._inner = inner

        def get_parameter(self, key):
            if key == "computation.operators.gradient.dx":
                return SimpleNamespace(bounds=(1.0e-10, 1.0e6), units="km", allowed_values=None)
            return self._inner.get_parameter(key)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        base_module,
        "get_global_manager",
        lambda *args, **kwargs: _KilometreManager(real_get_global_manager(*args, **kwargs)),
    )
    assert _gradient_operator(dx={"value": 500.0, "units": "m"}).dx == pytest.approx(0.5)


def test_yaml_bounds_apply_to_config_values(monkeypatch) -> None:
    """A narrower YAML bound rejects a config value (YAML is the bounds SSOT)."""
    from types import SimpleNamespace

    import relic.computation.operators.base as base_module

    real_get_global_manager = base_module.get_global_manager

    class _NarrowBoundsManager:
        def __init__(self, inner):
            self._inner = inner

        def get_parameter(self, key):
            if key == "computation.operators.gradient.dx":
                return SimpleNamespace(bounds=(2.0, 3.0), units="m", allowed_values=None)
            return self._inner.get_parameter(key)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        base_module,
        "get_global_manager",
        lambda *args, **kwargs: _NarrowBoundsManager(real_get_global_manager(*args, **kwargs)),
    )
    with pytest.raises(ValueError, match="outside bounds"):
        _gradient_operator(dx=1.0)
    assert _gradient_operator(dx=2.5).dx == 2.5


def _serve_gradient_boundary_allowed_values(monkeypatch, allowed_values) -> None:
    """Serve the YAML spec of the gradient boundary_method with other allowed_values."""
    import dataclasses

    from relic.parameters import get_global_manager

    manager = get_global_manager()
    original = manager.get_parameter

    def _patched(key, namespace=None):
        spec = original(key, namespace)
        if key == "computation.operators.gradient.boundary_method":
            return dataclasses.replace(spec, allowed_values=list(allowed_values))
        return spec

    monkeypatch.setattr(manager, "get_parameter", _patched)


def test_yaml_allowed_values_apply_to_config_choices(monkeypatch) -> None:
    """A narrower YAML allowed_values rejects a config choice (YAML is the choice SSOT)."""
    _serve_gradient_boundary_allowed_values(monkeypatch, ["zero", "periodic"])

    with pytest.raises(
        ValueError,
        match=r"boundary_method='extrapolate' not in the allowed values \(zero, periodic\)",
    ):
        _gradient_operator(boundary_method="extrapolate")
    assert _gradient_operator(boundary_method="periodic").boundary_method == "periodic"


def test_config_choice_outside_the_yaml_allowed_values_fails_at_construction() -> None:
    with pytest.raises(ValueError, match=r"boundary_method='reflect' not in the allowed values"):
        _gradient_operator(boundary_method="reflect")


def test_allowed_values_check_follows_the_validate_flag(monkeypatch) -> None:
    """validate_parameters=False skips the YAML checks (bounds and allowed_values) alike."""
    _serve_gradient_boundary_allowed_values(monkeypatch, ["zero", "periodic"])

    operator = _gradient_operator(validate_parameters=False, boundary_method="extrapolate")
    assert operator.boundary_method == "extrapolate"


def test_missing_key_is_not_replaced_by_a_default() -> None:
    with pytest.raises(KeyError, match="computation.operators.no_such_namespace.dx"):
        CentralDifferenceGradient1D(
            name="missing_key_probe", config={}, namespace="computation.operators.no_such_namespace"
        )


def _operator_expectations():
    from relic.computation.operators.diagnostics import (
        CFLCalculator,
        ConservationChecker,
        NonNegativeConstraint,
        StabilityChecker,
    )
    from relic.computation.operators.transport import (
        AdvectionDiffusionOperator,
        AdvectionOperator,
        DiffusionOperator,
    )

    diagnostics = "computation.operators.diagnostics"
    transport = "computation.operators.transport"
    return [
        (CFLCalculator, diagnostics, {"max_cfl_allowed": 1.0, "warn_threshold": 0.9}),
        # 0.5 (explicit-scheme limit) since 2026-09-23; previously 3000.0.
        (StabilityChecker, diagnostics, {"max_stability_allowed": 0.5, "warn_threshold": 0.9}),
        (ConservationChecker, diagnostics, {"tolerance": 1.0e-6, "strict_mode": False}),
        (NonNegativeConstraint, diagnostics, {"method": "clip", "small_value": 1.0e-12}),
        (
            DiffusionOperator,
            transport,
            {
                "diffusivity": 1.0e-5,
                "spatial_dims": 1,
                "boundary_method": "zero",
                "check_stability": True,
                "enforce_non_negative": True,
            },
        ),
        (
            AdvectionOperator,
            transport,
            {"spatial_dims": 1, "scheme": "upwind", "boundary_method": "extrapolate", "check_cfl": True},
        ),
        (
            AdvectionDiffusionOperator,
            transport,
            {
                "diffusivity": 1.0e-5,
                "spatial_dims": 1,
                "splitting_method": "symmetric",
                "advection_scheme": "upwind",
                "boundary_method": "zero",
                "check_stability": True,
                "enforce_non_negative": True,
            },
        ),
    ]


def test_operators_resolve_in_their_namespace_with_the_effective_values() -> None:
    """Each operator keeps its namespace and reads the values already in effect from YAML."""
    for operator_class, namespace, expected in _operator_expectations():
        operator = operator_class(name=f"{operator_class.__name__}_namespace_probe")
        assert operator.namespace == namespace, operator_class.__name__
        for attribute, value in expected.items():
            assert getattr(operator, attribute) == value, (operator_class.__name__, attribute)


def test_every_registered_operator_builds_through_the_factory() -> None:
    """All 15 registered operators build under strict lookups (no config needed)."""
    from relic.computation.operators import OperatorFactory
    from relic.computation.operators.base import OperatorMetadata
    from relic.computation.operators.registry import OperatorRegistry

    names = OperatorRegistry.list_operators()
    assert len(names) == 15
    for operator_name in names:
        operator = OperatorFactory.create(operator_name, name=f"{operator_name}_factory_probe")
        assert isinstance(operator.metadata, OperatorMetadata), operator_name


def test_laplacian_and_divergence_apply_match_the_functional_kernels() -> None:
    from relic.computation.operators.divergence import FiniteVolumeDivergence2D
    from relic.computation.operators.functional import (
        compute_divergence_2d,
        compute_laplacian_1d,
    )
    from relic.computation.operators.laplacian import FiniteDifferenceLaplacian1D

    field = torch.linspace(0.0, 1.0, 7, dtype=torch.float64) ** 2
    laplacian = FiniteDifferenceLaplacian1D(
        name="laplacian_probe", config={"dx": 0.5, "boundary_method": "zero"}, dtype=torch.float64
    )
    assert torch.equal(
        laplacian.apply(field),
        compute_laplacian_1d(field=field, dx=0.5, boundary_method="zero"),
    )

    flux_x = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    flux_y = flux_x.square()
    divergence = FiniteVolumeDivergence2D(
        name="divergence_probe",
        config={"dx": 2.0, "dy": 0.5, "boundary_method": "zero"},
        dtype=torch.float64,
    )
    assert torch.equal(
        divergence.apply(flux_x, flux_y),
        compute_divergence_2d(flux_x=flux_x, flux_y=flux_y, dx=2.0, dy=0.5, boundary_method="zero"),
    )


def test_custom_metadata_is_kept_by_laplacian_and_divergence() -> None:
    from relic.computation.operators.base import OperatorMetadata
    from relic.computation.operators.divergence import FiniteVolumeDivergence3D
    from relic.computation.operators.laplacian import FiniteDifferenceLaplacian2D

    custom = OperatorMetadata(name="custom", operator_type="custom", input_units="dimensionless", output_units="dimensionless", spatial_dimensions=0)
    assert FiniteDifferenceLaplacian2D(name="l2", metadata=custom).metadata is custom
    assert FiniteVolumeDivergence3D(name="d3", metadata=custom).metadata is custom


def test_stability_checker_limits_explicit_diffusion_at_one_half() -> None:
    """sigma = D dt / dx^2 (explicit diffusion): limit 0.5, warning at 0.9 * 0.5 = 0.45."""
    import warnings

    from relic.computation.operators.diagnostics import StabilityChecker

    checker = StabilityChecker(name="stability_limit_probe", dtype=torch.float64)
    one = 1.0
    with pytest.warns(UserWarning, match="exceeds maximum allowed 0.5"):
        sigma = checker.apply(torch.tensor([0.6], dtype=torch.float64), one, one)
    assert float(sigma.max()) == pytest.approx(0.6)
    with pytest.warns(UserWarning, match="exceeds warning threshold 0.450"):
        checker.apply(torch.tensor([0.46], dtype=torch.float64), one, one)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        checker.apply(torch.tensor([0.4], dtype=torch.float64), one, one)


def test_non_negative_constraint_zero_method_uses_its_yaml_floor() -> None:
    from relic.computation.operators.diagnostics import NonNegativeConstraint
    from relic.parameters import get_global_manager

    floor = float(get_global_manager().get("transport.numerical.non_negative_min_value", required=True))
    constraint = NonNegativeConstraint(name="zero_floor_probe", config={"method": "zero"})
    field = torch.tensor([-2.0, 4.0], dtype=torch.float32)
    assert torch.equal(constraint.apply(field), torch.tensor([floor, 4.0], dtype=torch.float32))


def test_positive_floor_keys_state_their_reader_and_difference() -> None:
    """The two clamp floors of apply_non_negative_constraint are documented as distinct."""
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    clip = manager.get_parameter("computation.operators.diagnostics.small_value").description
    zero = manager.get_parameter("transport.numerical.non_negative_min_value").description
    assert "NonNegativeConstraint" in clip and "'clip'" in clip
    assert "transport.numerical.non_negative_min_value" in clip
    assert "apply_non_negative_constraint" in zero and "'zero'" in zero
    assert "computation.operators.diagnostics.small_value" in zero
    assert "enforce_non_negative" in zero


def test_validate_parameters_flag_reaches_the_operator() -> None:
    from relic.computation.operators.transport import DiffusionOperator

    operator = DiffusionOperator(name="validate_flag_probe", validate_parameters=False)
    assert operator.validate_parameters is False


# ---------------------------------------------------------------------------
# Runtime keywords and the computation dtype. A misspelled runtime override is
# rejected instead of being ignored in favour of the configured value, and an
# operator that casts its inputs to its own dtype resolves dtype=None to the
# OperatorBase default float32 instead of computing integer input in integer
# arithmetic (silent truncation).
# ---------------------------------------------------------------------------


def _runtime_keyword_cases():
    from relic.computation.operators.diagnostics import (
        CFLCalculator,
        ConservationChecker,
        NonNegativeConstraint,
        StabilityChecker,
    )
    from relic.computation.operators.divergence import (
        FiniteVolumeDivergence2D,
        FiniteVolumeDivergence3D,
    )
    from relic.computation.operators.laplacian import (
        FiniteDifferenceLaplacian1D,
        FiniteDifferenceLaplacian2D,
        FiniteDifferenceLaplacian3D,
    )
    from relic.computation.operators.transport import (
        AdvectionDiffusionOperator,
        AdvectionOperator,
        DiffusionOperator,
    )

    field_1d = torch.tensor([[0.0, 1.0, 3.0, 2.0, 1.0]], dtype=torch.float64)
    field_2d = torch.ones(1, 3, 3, dtype=torch.float64)
    field_3d = torch.ones(1, 3, 3, 3, dtype=torch.float64)
    velocity = torch.full_like(field_1d, 0.5)
    return [
        (FiniteVolumeDivergence2D, (field_2d, field_2d)),
        (FiniteVolumeDivergence3D, (field_3d, field_3d, field_3d)),
        (FiniteDifferenceLaplacian1D, (field_1d,)),
        (FiniteDifferenceLaplacian2D, (field_2d,)),
        (FiniteDifferenceLaplacian3D, (field_3d,)),
        (DiffusionOperator, (field_1d, 1.0, 0.1)),
        (AdvectionOperator, (field_1d, velocity, 1.0, 0.1)),
        (AdvectionDiffusionOperator, (field_1d, velocity, 1.0, 0.1)),
        (CFLCalculator, (velocity, 1.0, 0.1)),
        (StabilityChecker, (torch.full_like(field_1d, 0.01), 1.0, 0.1)),
        (ConservationChecker, (field_1d, field_1d)),
        (NonNegativeConstraint, (field_1d,)),
    ]


@pytest.mark.parametrize(
    ("operator_class", "arguments"),
    _runtime_keyword_cases(),
    ids=lambda value: getattr(value, "__name__", "args"),
)
def test_unknown_runtime_keyword_is_rejected_by_every_operator(operator_class, arguments) -> None:
    operator = operator_class(name=f"{operator_class.__name__}_keyword_probe", dtype=torch.float64)

    with pytest.raises(TypeError, match="unexpected"):
        operator(*arguments, misspelled_override=2.0)


def _dtype_casting_operator_classes():
    from relic.computation.operators.diagnostics import (
        CFLCalculator,
        ConservationChecker,
        NonNegativeConstraint,
        StabilityChecker,
    )
    from relic.computation.operators.divergence import (
        FiniteVolumeDivergence2D,
        FiniteVolumeDivergence3D,
    )
    from relic.computation.operators.laplacian import (
        FiniteDifferenceLaplacian1D,
        FiniteDifferenceLaplacian2D,
        FiniteDifferenceLaplacian3D,
    )

    return [
        FiniteVolumeDivergence2D,
        FiniteVolumeDivergence3D,
        FiniteDifferenceLaplacian1D,
        FiniteDifferenceLaplacian2D,
        FiniteDifferenceLaplacian3D,
        CFLCalculator,
        StabilityChecker,
        ConservationChecker,
        NonNegativeConstraint,
        CentralDifferenceGradient1D,
    ]


@pytest.mark.parametrize(
    "operator_class",
    _dtype_casting_operator_classes(),
    ids=lambda value: value.__name__,
)
def test_missing_dtype_resolves_to_the_operator_base_default(operator_class) -> None:
    operator = operator_class(name=f"{operator_class.__name__}_dtype_probe", dtype=None)

    assert operator.dtype == torch.float32


@pytest.mark.parametrize(
    "operator_class",
    _dtype_casting_operator_classes(),
    ids=lambda value: value.__name__,
)
def test_non_floating_operator_dtype_is_rejected(operator_class) -> None:
    with pytest.raises(TypeError, match="floating point or complex"):
        operator_class(name=f"{operator_class.__name__}_int_probe", dtype=torch.int64)


def test_laplacian_of_an_integer_field_is_not_truncated() -> None:
    """d2f/dx2 of f = [0, 1, 3, 6, 10] with dx = 2 m and the zero closure.

    Interior: (f[i+1] - 2 f[i] + f[i-1]) / dx^2 = 1/4; boundaries (reflected
    ghost cells): 2 (f[1] - f[0]) / dx^2 = 0.5 and 2 (f[3] - f[4]) / dx^2 = -2.
    """
    from relic.computation.operators.laplacian import FiniteDifferenceLaplacian1D

    operator = FiniteDifferenceLaplacian1D(
        name="integer_laplacian_probe",
        config={"dx": 2.0, "boundary_method": "zero"},
        dtype=None,
    )
    result = operator(torch.tensor([[0, 1, 3, 6, 10]], dtype=torch.int64))

    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result, torch.tensor([[0.5, 0.25, 0.25, 0.25, -2.0]], dtype=torch.float32)
    )


def test_divergence_of_an_integer_flux_is_not_truncated() -> None:
    from relic.computation.operators.divergence import FiniteVolumeDivergence2D
    from relic.computation.operators.functional import compute_divergence_2d

    operator = FiniteVolumeDivergence2D(
        name="integer_divergence_probe",
        config={"dx": 4.0, "dy": 4.0, "boundary_method": "extrapolate"},
        dtype=None,
    )
    flux = torch.arange(16, dtype=torch.int64).reshape(1, 4, 4)
    result = operator(flux, flux)

    expected = compute_divergence_2d(
        flux_x=flux.to(torch.float32),
        flux_y=flux.to(torch.float32),
        dx=4.0,
        dy=4.0,
        boundary_method="extrapolate",
    )
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# The YAML allowed_values of every operator choice key are implemented: each
# accepted value constructs the consuming operator (OperatorBase checks it
# against the spec) and runs through the functional kernel it selects.
# ---------------------------------------------------------------------------


def _probe_field() -> torch.Tensor:
    return torch.tensor([[0.0, 1.0, 3.0, 2.0, 1.0]], dtype=torch.float64)


def _probe_gradient_boundary(value: str) -> None:
    CentralDifferenceGradient1D(
        name="allowed_gradient_probe",
        config={"dx": 1.0, "boundary_method": value, "order": 2},
        dtype=torch.float64,
    )(_probe_field())


def _probe_divergence_boundary(value: str) -> None:
    from relic.computation.operators.divergence import FiniteVolumeDivergence2D

    flux = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4)
    FiniteVolumeDivergence2D(
        name="allowed_divergence_probe",
        config={"dx": 1.0, "dy": 1.0, "boundary_method": value},
        dtype=torch.float64,
    )(flux, flux)


def _probe_laplacian_boundary(value: str) -> None:
    from relic.computation.operators.laplacian import FiniteDifferenceLaplacian1D

    FiniteDifferenceLaplacian1D(
        name="allowed_laplacian_probe",
        config={"dx": 1.0, "boundary_method": value},
        dtype=torch.float64,
    )(_probe_field())


def _advection_diffusion(config: dict) -> object:
    from relic.computation.operators.transport import AdvectionDiffusionOperator

    resolved = {
        "diffusivity": 0.005,
        "spatial_dims": 1,
        "splitting_method": "sequential",
        "advection_scheme": "upwind",
        "boundary_method": "periodic",
        "check_stability": True,
        "enforce_non_negative": False,
        **config,
    }
    return AdvectionDiffusionOperator(name="allowed_transport_probe", config=resolved)


def _advection(config: dict) -> object:
    from relic.computation.operators.transport import AdvectionOperator

    resolved = {
        "spatial_dims": 1,
        "scheme": "upwind",
        "boundary_method": "periodic",
        "check_cfl": True,
        **config,
    }
    return AdvectionOperator(name="allowed_advection_probe", config=resolved)


def _probe_transport_boundary(value: str) -> None:
    """transport.boundary_method feeds DiffusionOperator and AdvectionDiffusionOperator.

    Dirichlet closures need explicit boundary states for both kernels; the
    open advective inflow needs one for advection.
    """
    from relic.computation.operators.transport import DiffusionOperator

    field = _probe_field()
    state = torch.ones_like(field)
    DiffusionOperator(
        name="allowed_diffusion_probe",
        config={
            "diffusivity": 0.005,
            "spatial_dims": 1,
            "boundary_method": value,
            "check_stability": True,
            "enforce_non_negative": False,
        },
    )(field, 1.0, 0.1, **({"boundary_state": state} if value == "dirichlet" else {}))
    states = {}
    if value in ("dirichlet", "open"):
        states["advection_boundary_state"] = state
    if value == "dirichlet":
        states["diffusion_boundary_state"] = state
    _advection_diffusion({"boundary_method": value})(
        field, torch.full_like(field, 0.5), 1.0, 0.1, time_integration_method="euler", **states
    )


def _probe_advection_boundary(value: str) -> None:
    field = _probe_field()
    states = {"boundary_state": torch.ones_like(field)} if value in ("dirichlet", "open") else {}
    _advection({"boundary_method": value})(field, torch.full_like(field, 0.5), 1.0, 0.1, **states)


def _probe_advection_scheme(value: str) -> None:
    """AdvectionOperator scheme; central FTCS is unstable with euler, so it steps implicitly."""
    field = _probe_field()
    method = "implicit_midpoint" if value == "central" else "euler"
    _advection({"scheme": value, "time_integration_method": method})(
        field, torch.full_like(field, 0.5), 1.0, 0.1
    )


def _probe_advection_diffusion_scheme(value: str) -> None:
    field = _probe_field()
    method = "implicit_midpoint" if value == "central" else "euler"
    _advection_diffusion({"advection_scheme": value, "time_integration_method": method})(
        field, torch.full_like(field, 0.5), 1.0, 0.1
    )


def _probe_splitting(value: str) -> None:
    field = _probe_field()
    _advection_diffusion({"splitting_method": value})(
        field, torch.full_like(field, 0.5), 1.0, 0.1, time_integration_method="euler"
    )


def _probe_non_negative_method(value: str) -> None:
    from relic.computation.operators.diagnostics import NonNegativeConstraint

    NonNegativeConstraint(
        name="allowed_constraint_probe",
        config={"method": value, "small_value": 1.0e-12},
        dtype=torch.float64,
    )(torch.tensor([[-1.0, 0.5]], dtype=torch.float64))


_CHOICE_PROBES = {
    "computation.operators.gradient.boundary_method": _probe_gradient_boundary,
    "computation.operators.divergence.boundary_method": _probe_divergence_boundary,
    "computation.operators.laplacian.boundary_method": _probe_laplacian_boundary,
    "computation.operators.transport.boundary_method": _probe_transport_boundary,
    "computation.operators.transport.advection.boundary_method": _probe_advection_boundary,
    "computation.operators.transport.scheme": _probe_advection_scheme,
    "computation.operators.transport.advection_scheme": _probe_advection_diffusion_scheme,
    "computation.operators.transport.splitting_method": _probe_splitting,
    "computation.operators.diagnostics.method": _probe_non_negative_method,
}


def test_choice_probes_cover_every_operator_choice_key() -> None:
    from pathlib import Path

    import yaml

    import relic

    database = (
        Path(relic.__file__).parent / "parameters" / "database" / "relic"
        / "computation" / "operators.yaml"
    )
    entries = yaml.safe_load(database.read_text(encoding="utf-8"))["parameters"]
    choice_keys = {
        key for key, entry in entries.items()
        if isinstance(entry, dict) and entry.get("allowed_values")
    }
    assert choice_keys == set(_CHOICE_PROBES)


@pytest.mark.parametrize("key", sorted(_CHOICE_PROBES))
def test_every_yaml_allowed_choice_is_implemented_by_its_operator(key: str) -> None:
    from relic.parameters import get_global_manager

    spec = get_global_manager().get_parameter(key)
    assert spec is not None and spec.allowed_values, key
    for value in spec.allowed_values:
        _CHOICE_PROBES[key](value)


# ---------------------------------------------------------------------------
# Round 5: every operator declares the units it maps and the dimensions it
# supports; the base class invents no 'dimensionless' / 0 declaration.
# ---------------------------------------------------------------------------


EXPECTED_OPERATOR_UNITS = {
    "gradient.central_difference_1d": ("[field_units]", "[field_units] m-1", 1),
    "gradient.central_difference_2d": ("[field_units]", "[field_units] m-1", 2),
    "gradient.central_difference_3d": ("[field_units]", "[field_units] m-1", 3),
    "divergence.finite_volume_2d": ("[flux_units]", "[flux_units] m-1", 2),
    "divergence.finite_volume_3d": ("[flux_units]", "[flux_units] m-1", 3),
    "laplacian.finite_difference_1d": ("[field_units]", "[field_units] m-2", 1),
    "laplacian.finite_difference_2d": ("[field_units]", "[field_units] m-2", 2),
    "laplacian.finite_difference_3d": ("[field_units]", "[field_units] m-2", 3),
    "diagnostics.cfl_calculator": ("m s-1", "dimensionless", 0),
    "diagnostics.stability_checker": ("m2 s-1", "dimensionless", 0),
    "diagnostics.conservation_checker": ("kg m-3", "dimensionless", 0),
    "diagnostics.non_negative_constraint": ("[field_units]", "[field_units]", 0),
}


@pytest.mark.parametrize("operator_name", sorted(EXPECTED_OPERATOR_UNITS))
def test_registered_operator_declares_its_documented_units(operator_name: str) -> None:
    from relic.computation.operators import OperatorFactory

    operator = OperatorFactory.create(operator_name, name=f"{operator_name}_units_probe")
    metadata = operator.get_metadata()
    expected = EXPECTED_OPERATOR_UNITS[operator_name]
    assert (metadata.input_units, metadata.output_units, metadata.spatial_dimensions) == expected


def test_operator_without_declared_metadata_is_refused() -> None:
    from relic.computation.operators.base import OperatorBase

    class _UndeclaredOperator(OperatorBase):
        operator_type = "probe"

        def apply(self, field):
            return field

    operator = _UndeclaredOperator(name="undeclared_probe")
    with pytest.raises(TypeError, match="OperatorMetadata"):
        operator.get_metadata()
