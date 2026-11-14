"""Direct L1 behavior tests for registered finite-difference gradients.

References:
    LeVeque (2007); Fornberg (1988).
Mathematical expression:
    grad(f) = (df/dx, df/dy, df/dz), with centered finite differences.
Parameter meanings:
    dx, dy, dz are positive grid spacings and boundary_method selects the
    discrete boundary closure.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from relic.computation.operators import OperatorFactory
from relic.computation.operators.base import OperatorBase, OperatorMetadata
from relic.computation.operators.functional import (
    compute_gradient_1d,
    compute_gradient_2d,
    compute_gradient_3d,
)
from relic.computation.operators.gradient import (
    CentralDifferenceGradient1D,
    CentralDifferenceGradient2D,
    CentralDifferenceGradient3D,
)
from relic.computation.operators.registry import OperatorRegistry


GradientClass = type[OperatorBase]


CASES: tuple[
    tuple[str, GradientClass, dict[str, object], tuple[int, ...], int],
    ...,
] = (
    (
        "gradient.central_difference_1d",
        CentralDifferenceGradient1D,
        {"dx": 0.5, "boundary_method": "extrapolate", "order": 2},
        (1, 6),
        1,
    ),
    (
        "gradient.central_difference_2d",
        CentralDifferenceGradient2D,
        {"dx": 0.5, "dy": 2.0, "boundary_method": "extrapolate"},
        (1, 5, 6),
        2,
    ),
    (
        "gradient.central_difference_3d",
        CentralDifferenceGradient3D,
        {
            "dx": 0.5,
            "dy": 1.5,
            "dz": 2.0,
            "boundary_method": "extrapolate",
        },
        (1, 5, 5, 6),
        3,
    ),
)


def _construct(
    operator_class: GradientClass,
    config: dict[str, object],
    *,
    dtype: torch.dtype | None = torch.float64,
    metadata: OperatorMetadata | None = None,
    validate_parameters: bool = True,
) -> OperatorBase:
    return operator_class(
        name=f"test_{operator_class.__name__}",
        config=config,
        metadata=metadata,
        dtype=dtype,
        validate_parameters=validate_parameters,
    )


def _linear_field_and_slopes(
    dimensions: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, tuple[float, ...]]:
    if dimensions == 1:
        x = torch.arange(6, dtype=dtype) * 0.5
        return (2.0 * x + 1.0).unsqueeze(0), (2.0,)
    if dimensions == 2:
        y, x = torch.meshgrid(
            torch.arange(5, dtype=dtype) * 2.0,
            torch.arange(6, dtype=dtype) * 0.5,
            indexing="ij",
        )
        return (2.0 * x + 3.0 * y + 1.0).unsqueeze(0), (2.0, 3.0)
    z, y, x = torch.meshgrid(
        torch.arange(5, dtype=dtype) * 2.0,
        torch.arange(5, dtype=dtype) * 1.5,
        torch.arange(6, dtype=dtype) * 0.5,
        indexing="ij",
    )
    return (2.0 * x + 3.0 * y + 5.0 * z + 1.0).unsqueeze(0), (2.0, 3.0, 5.0)


@pytest.mark.parametrize(
    ("registry_name", "operator_class", "config", "_shape", "dimensions"),
    CASES,
)
def test_registered_gradient_classes_are_factory_creatable(
    registry_name: str,
    operator_class: GradientClass,
    config: dict[str, object],
    _shape: tuple[int, ...],
    dimensions: int,
) -> None:
    entry = OperatorRegistry.get_entry(registry_name)
    operator = OperatorFactory.create(
        operator_name=registry_name,
        name=f"factory_gradient_{dimensions}d",
        config=config,
        dtype=torch.float64,
        validate_parameters=False,
    )

    assert entry.operator_class is operator_class
    assert entry.operator_type == "gradient"
    assert entry.category == "spatial"
    assert entry.metadata is not None
    assert isinstance(operator, operator_class)


@pytest.mark.parametrize(
    ("_registry_name", "operator_class", "config", "_shape", "dimensions"),
    CASES,
)
def test_anisotropic_linear_fields_preserve_component_order(
    _registry_name: str,
    operator_class: GradientClass,
    config: dict[str, object],
    _shape: tuple[int, ...],
    dimensions: int,
) -> None:
    operator = _construct(operator_class, config)
    field, slopes = _linear_field_and_slopes(dimensions)

    result = operator(field)
    components = (result,) if isinstance(result, torch.Tensor) else result

    assert len(components) == dimensions
    for component, slope in zip(components, slopes, strict=True):
        assert component.shape == field.shape
        assert component.dtype == torch.float64
        assert component.device == field.device
        torch.testing.assert_close(component, torch.full_like(field, slope))


@pytest.mark.parametrize("boundary_method", ("zero", "extrapolate", "periodic"))
def test_1d_wrapper_matches_functional_ssot_with_runtime_overrides(
    boundary_method: str,
) -> None:
    operator = _construct(
        CentralDifferenceGradient1D,
        {"dx": 1.0, "boundary_method": "zero", "order": 2},
    )
    field = torch.tensor(
        [[0.0, 1.0, 4.0, 9.0, 16.0, 25.0]],
        dtype=torch.float64,
    )

    actual = operator.apply(
        field,
        dx=0.5,
        boundary_method=boundary_method,
        order=4 if boundary_method == "periodic" else 2,
    )
    expected = compute_gradient_1d(
        field,
        dx=0.5,
        boundary_method=boundary_method,
        order=4 if boundary_method == "periodic" else 2,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("operator_class", "config", "functional"),
    (
        (
            CentralDifferenceGradient2D,
            {"dx": 1.0, "dy": 1.0, "boundary_method": "zero"},
            compute_gradient_2d,
        ),
        (
            CentralDifferenceGradient3D,
            {
                "dx": 1.0,
                "dy": 1.0,
                "dz": 1.0,
                "boundary_method": "zero",
            },
            compute_gradient_3d,
        ),
    ),
)
def test_multidimensional_wrappers_match_functional_ssot(
    operator_class: GradientClass,
    config: dict[str, object],
    functional: Callable[..., tuple[torch.Tensor, ...]],
) -> None:
    dimensions = 2 if operator_class is CentralDifferenceGradient2D else 3
    field, _ = _linear_field_and_slopes(dimensions)
    operator = _construct(operator_class, config)
    runtime = {
        "dx": 0.5,
        "dy": 2.0 if dimensions == 2 else 1.5,
        "boundary_method": "periodic",
    }
    if dimensions == 3:
        runtime["dz"] = 2.0

    actual = operator.apply(field, **runtime)
    expected = functional(field, **runtime)

    assert len(actual) == len(expected)
    for actual_component, expected_component in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_component, expected_component)


@pytest.mark.parametrize(
    ("_registry_name", "operator_class", "config", "_shape", "dimensions"),
    CASES,
)
def test_default_metadata_declares_dimension_and_stencil(
    _registry_name: str,
    operator_class: GradientClass,
    config: dict[str, object],
    _shape: tuple[int, ...],
    dimensions: int,
) -> None:
    operator = _construct(operator_class, config)
    metadata = operator.get_metadata()

    assert metadata.operator_type == "gradient"
    assert metadata.spatial_dimensions == dimensions
    assert metadata.requires_grad is True
    assert metadata.mathematical_expression
    assert metadata.references
    assert metadata.spatial_requirements is not None
    assert metadata.spatial_requirements.stencil_radius == 1
    assert metadata.spatial_requirements.halo_depth == 1


def test_fourth_order_metadata_declares_radius_two() -> None:
    operator = _construct(
        CentralDifferenceGradient1D,
        {"dx": 1.0, "boundary_method": "periodic", "order": 4},
    )

    assert operator.metadata.spatial_requirements is not None
    assert operator.metadata.spatial_requirements.stencil_radius == 2
    assert operator.metadata.spatial_requirements.halo_depth == 2
    assert "12" in operator.metadata.mathematical_expression


def test_custom_metadata_is_preserved_by_constructor_and_factory() -> None:
    custom = OperatorMetadata(
        name="custom_gradient",
        operator_type="gradient",
        description="Caller-owned metadata",
        spatial_dimensions=1, input_units="dimensionless", output_units="dimensionless",
    )
    direct = _construct(
        CentralDifferenceGradient1D,
        {"dx": 1.0, "boundary_method": "zero", "order": 2},
        metadata=custom,
    )
    factory = OperatorFactory.create(
        operator_name="gradient.central_difference_1d",
        name="custom_factory_gradient",
        config={"dx": 1.0, "boundary_method": "zero", "order": 2},
        metadata=custom,
        validate_parameters=False,
    )

    assert direct.metadata is custom
    assert factory.metadata is custom


def test_default_dtype_converts_integer_field_without_truncation() -> None:
    operator = _construct(
        CentralDifferenceGradient1D,
        {"dx": 2.0, "boundary_method": "extrapolate", "order": 2},
        dtype=None,
    )
    field = torch.tensor([[0, 1, 3, 6, 10]], dtype=torch.int64)

    result = operator(field)

    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result,
        torch.tensor([[0.5, 0.75, 1.25, 1.75, 2.0]], dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("_registry_name", "operator_class", "config", "shape", "_dimensions"),
    CASES,
)
def test_gradient_wrappers_are_autograd_differentiable(
    _registry_name: str,
    operator_class: GradientClass,
    config: dict[str, object],
    shape: tuple[int, ...],
    _dimensions: int,
) -> None:
    operator = _construct(operator_class, config)
    field = torch.randn(shape, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(operator.apply, (field,), fast_mode=True)


def test_standard_module_to_updates_output_dtype_and_diagnostics() -> None:
    operator = _construct(
        CentralDifferenceGradient1D,
        {"dx": 1.0, "boundary_method": "extrapolate", "order": 2},
        dtype=torch.float32,
    )

    operator.to(device=torch.device("cpu"), dtype=torch.float64)
    result = operator(torch.arange(5, dtype=torch.float32).reshape(1, -1))

    assert result.dtype == torch.float64
    assert operator.get_diagnostics()["dtype"] == "torch.float64"
    assert operator.get_diagnostics()["device"] == "cpu"


def test_module_call_equals_apply_and_keeps_no_counter() -> None:
    """I1: the module call is apply(); no per-call counter is kept."""
    operator = _construct(
        CentralDifferenceGradient1D,
        {"dx": 1.0, "boundary_method": "extrapolate", "order": 2},
    )
    field = torch.arange(5, dtype=torch.float64).reshape(1, -1)

    torch.testing.assert_close(operator(field), operator.apply(field))

    assert "apply_count" not in operator.get_diagnostics()


@pytest.mark.parametrize(
    ("operator_class", "config", "field", "message"),
    (
        (
            CentralDifferenceGradient1D,
            {"dx": 1.0, "boundary_method": "zero", "order": 2},
            torch.ones(3),
            "at least 2 dimensions",
        ),
        (
            CentralDifferenceGradient2D,
            {"dx": 1.0, "dy": 1.0, "boundary_method": "zero"},
            torch.ones(1, 2, 3),
            "at least 3 grid points",
        ),
        (
            CentralDifferenceGradient3D,
            {
                "dx": 1.0,
                "dy": 1.0,
                "dz": 1.0,
                "boundary_method": "zero",
            },
            torch.ones(1, 3, 3),
            "4 dimensions",
        ),
    ),
)
def test_invalid_rank_or_axis_extent_fails_fast(
    operator_class: GradientClass,
    config: dict[str, object],
    field: torch.Tensor,
    message: str,
) -> None:
    operator = _construct(operator_class, config)

    with pytest.raises(ValueError, match=message):
        operator(field)


@pytest.mark.parametrize("dx", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_constructor_spacing_fails_fast(dx: float) -> None:
    with pytest.raises(ValueError, match="outside bounds"):
        _construct(
            CentralDifferenceGradient1D,
            {"dx": dx, "boundary_method": "zero", "order": 2},
        )


@pytest.mark.parametrize(
    ("order", "message"),
    (
        # Inside the YAML bounds [1, 4] of computation.operators.gradient.order:
        # rejected by the operator's own supported-order check.
        (1, "order must be 2 or 4"),
        (3, "order must be 2 or 4"),
        # Outside the YAML bounds: rejected first by the YAML bounds check.
        (5, "outside bounds"),
    ),
)
def test_unsupported_constructor_order_fails_fast(order: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _construct(
            CentralDifferenceGradient1D,
            {"dx": 1.0, "boundary_method": "zero", "order": order},
        )


def test_invalid_boundary_and_runtime_spacing_fail_fast() -> None:
    operator = _construct(
        CentralDifferenceGradient1D,
        {"dx": 1.0, "boundary_method": "zero", "order": 2},
    )
    field = torch.arange(5, dtype=torch.float64).reshape(1, -1)

    with pytest.raises(ValueError, match="boundary method"):
        operator(field, boundary_method="reflect")
    with pytest.raises(ValueError, match="grid spacing"):
        operator(field, dx=0.0)


@pytest.mark.parametrize(
    ("operator_class", "config", "field"),
    (
        (
            CentralDifferenceGradient1D,
            {"dx": 1.0, "boundary_method": "zero", "order": 2},
            torch.ones(1, 5),
        ),
        (
            CentralDifferenceGradient2D,
            {"dx": 1.0, "dy": 1.0, "boundary_method": "zero"},
            torch.ones(1, 3, 3),
        ),
        (
            CentralDifferenceGradient3D,
            {
                "dx": 1.0,
                "dy": 1.0,
                "dz": 1.0,
                "boundary_method": "zero",
            },
            torch.ones(1, 3, 3, 3),
        ),
    ),
)
def test_unknown_runtime_keyword_is_rejected(
    operator_class: GradientClass,
    config: dict[str, object],
    field: torch.Tensor,
) -> None:
    operator = _construct(operator_class, config)

    with pytest.raises(TypeError, match="unexpected runtime parameter"):
        operator(field, misspelled_spacing=2.0)


def test_non_tensor_and_nonfinite_fields_fail_fast() -> None:
    operator = _construct(
        CentralDifferenceGradient1D,
        {"dx": 1.0, "boundary_method": "zero", "order": 2},
    )

    with pytest.raises(TypeError, match="torch.Tensor"):
        operator([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        operator(torch.tensor([[0.0, float("nan"), 2.0]]))


@pytest.mark.parametrize(
    ("operator_class", "config", "field"),
    (
        (
            CentralDifferenceGradient1D,
            {"dx": 1.0, "boundary_method": "reflect", "order": 2},
            torch.arange(5, dtype=torch.float64).reshape(1, -1),
        ),
        (
            CentralDifferenceGradient2D,
            {"dx": 1.0, "dy": 1.0, "boundary_method": "reflect"},
            torch.ones(1, 3, 3, dtype=torch.float64),
        ),
        (
            CentralDifferenceGradient3D,
            {"dx": 1.0, "dy": 1.0, "dz": 1.0, "boundary_method": "reflect"},
            torch.ones(1, 3, 3, 3, dtype=torch.float64),
        ),
    ),
)
def test_boundary_choices_come_from_the_yaml_spec_not_a_code_list(
    monkeypatch,
    operator_class: GradientClass,
    config: dict[str, object],
    field: torch.Tensor,
) -> None:
    """The configured choice is checked only against the YAML allowed_values.

    With the spec widened to a value no kernel implements, the operator
    accepts the configuration (no second list in the operator) and the
    functional kernel, which owns the implemented closures, rejects it.
    """
    import dataclasses

    from relic.parameters import get_global_manager

    manager = get_global_manager()
    original = manager.get_parameter

    def _patched(key, namespace=None):
        spec = original(key, namespace)
        if key == "computation.operators.gradient.boundary_method":
            return dataclasses.replace(
                spec, allowed_values=[*spec.allowed_values, "reflect"]
            )
        return spec

    monkeypatch.setattr(manager, "get_parameter", _patched)

    operator = _construct(operator_class, config)
    assert operator.boundary_method == "reflect"
    with pytest.raises(ValueError, match="Gradient boundary method must be"):
        operator(field)
