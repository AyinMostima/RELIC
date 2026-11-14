"""Tests for DEC complex bundles."""

from dataclasses import replace

import pytest
import torch


def _strict_bundle(**overrides):
    from relic.computation.operators.dec import DECComplexBundle

    values = {
        "complex_id": "strict_interval",
        "dimension": 1,
        "n_cells": {0: 2, 1: 1},
        "D": {0: torch.tensor([[-1.0, 1.0]], dtype=torch.float64)},
        "boundary_masks": {
            0: torch.tensor([True, True], dtype=torch.bool),
            1: torch.tensor([False], dtype=torch.bool),
        },
        "orientation": {
            0: torch.tensor([1, 1], dtype=torch.int8),
            1: torch.tensor([1], dtype=torch.int8),
        },
        "cell_measures": {
            0: torch.tensor([1.0, 1.0], dtype=torch.float64),
            1: torch.tensor([1.0], dtype=torch.float64),
        },
        "dual_measures": {
            0: torch.tensor([0.5, 0.5], dtype=torch.float64),
            1: torch.tensor([1.0], dtype=torch.float64),
        },
        "orientation_convention": "outward_positive",
    }
    values.update(overrides)
    return DECComplexBundle(**values)


def _strict_triangle_bundle(**overrides):
    from relic.computation.operators.dec import DECComplexBundle

    values = {
        "complex_id": "strict_triangle",
        "dimension": 2,
        "n_cells": {0: 3, 1: 3, 2: 1},
        "D": {
            0: torch.tensor(
                [
                    [-1.0, 1.0, 0.0],
                    [0.0, -1.0, 1.0],
                    [1.0, 0.0, -1.0],
                ],
                dtype=torch.float64,
            ),
            1: torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64),
        },
        "boundary_masks": {
            0: torch.tensor([True, True, True], dtype=torch.bool),
            1: torch.tensor([True, True, True], dtype=torch.bool),
            2: torch.tensor([False], dtype=torch.bool),
        },
        "orientation": {
            0: torch.tensor([1, 1, 1], dtype=torch.int8),
            1: torch.tensor([1, 1, 1], dtype=torch.int8),
            2: torch.tensor([1], dtype=torch.int8),
        },
        "cell_measures": {
            0: torch.ones(3, dtype=torch.float64),
            1: torch.ones(3, dtype=torch.float64),
            2: torch.ones(1, dtype=torch.float64),
        },
        "dual_measures": {
            0: torch.ones(3, dtype=torch.float64),
            1: torch.ones(3, dtype=torch.float64),
            2: torch.ones(1, dtype=torch.float64),
        },
        "orientation_convention": "outward_positive",
    }
    values.update(overrides)
    return DECComplexBundle(**values)


def _typed_interval_pair(*, alpha_space=None, chain_space=None, chain_dtype=None):
    from relic.computation.operators.dec import (
        ChainSpaceSpec,
        CochainSpaceSpec,
        TypedChain,
        TypedCochain,
    )

    if alpha_space is None:
        alpha_space = CochainSpaceSpec(
            complex_id="strict_interval",
            degree=0,
            primality="primal",
            n_dofs=2,
            units="J mol-1",
            extensivity="intensive",
            physical_domain="test_domain",
            semantic_name="test_effort",
            orientation_convention="outward_positive",
            boundary_role="boundary",
        )
    if chain_space is None:
        chain_space = ChainSpaceSpec(
            complex_id="strict_interval",
            degree=1,
            primality="primal",
            n_dofs=1,
            coefficient_units="1",
            physical_domain="test_domain",
            semantic_name="test_integration_chain",
            orientation_convention="outward_positive",
            boundary_role="interior",
        )
    alpha = TypedCochain(
        name="alpha",
        values=torch.tensor([2.0, 5.0], dtype=torch.float64),
        space=alpha_space,
    )
    chain = TypedChain(
        name="c",
        coefficients=torch.tensor(
            [3.0],
            dtype=chain_dtype or torch.float64,
        ),
        space=chain_space,
    )
    return alpha, chain


def test_dec_complex_validates_boundary_of_boundary_zero():
    from relic.computation.operators.dec import DECComplexBundle

    d0 = torch.tensor(
        [
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    d1 = torch.tensor([[1.0, 1.0, 1.0]])
    bundle = DECComplexBundle(
        complex_id="soil_column",
        dimension=2,
        n_cells={0: 3, 1: 3, 2: 1},
        D={0: d0, 1: d1},
        boundary_masks={},
        orientation={},
        cell_measures={},
        dual_measures={},
        orientation_convention="outward_positive",
    )

    residuals = bundle.validate_complex(tol=0.0)

    assert residuals[(1, 0)] == 0.0


def test_dec_complex_rejects_nonzero_composition():
    from relic.computation.operators.dec import DECComplexBundle

    bundle = DECComplexBundle(
        complex_id="bad_complex",
        dimension=2,
        n_cells={0: 2, 1: 2, 2: 1},
        D={
            0: torch.tensor([[-1.0, 1.0], [0.0, -1.0]]),
            1: torch.tensor([[1.0, 1.0]]),
        },
        boundary_masks={},
        orientation={},
        cell_measures={},
        dual_measures={},
        orientation_convention="outward_positive",
    )

    with pytest.raises(ValueError, match="D\\[1\\] @ D\\[0\\]"):
        bundle.validate_complex(tol=0.0)


def test_dec_complex_rejects_derivative_shape_mismatch():
    from relic.computation.operators.dec import DECComplexBundle

    with pytest.raises(ValueError, match="shape"):
        DECComplexBundle(
            complex_id="shape_mismatch",
            dimension=1,
            n_cells={0: 3, 1: 2},
            D={0: torch.ones(3, 3)},
            boundary_masks={},
            orientation={},
            cell_measures={},
            dual_measures={},
            orientation_convention="outward_positive",
        )


def test_dec_complex_strict_mode_rejects_incomplete_derivative_sequence():
    from relic.computation.operators.dec import DECComplexBundle

    bundle = DECComplexBundle(
        complex_id="incomplete_3d_complex",
        dimension=3,
        n_cells={0: 1, 1: 1, 2: 1, 3: 1},
        D={0: torch.zeros((1, 1))},
        boundary_masks={},
        orientation={},
        cell_measures={},
        dual_measures={},
        orientation_convention="outward_positive",
    )

    with pytest.raises(ValueError, match="complete derivative sequence"):
        bundle.validate_complex(tol=0.0, strict=True)


def test_dec_complex_strict_mode_rejects_missing_support_data():
    from relic.computation.operators.dec import DECComplexBundle

    bundle = DECComplexBundle(
        complex_id="missing_support_data",
        dimension=1,
        n_cells={0: 2, 1: 1},
        D={0: torch.tensor([[-1.0, 1.0]])},
        boundary_masks={},
        orientation={},
        cell_measures={},
        dual_measures={},
        orientation_convention="outward_positive",
    )

    with pytest.raises(ValueError, match="boundary_masks"):
        bundle.validate_complex(tol=0.0, strict=True)


def test_dec_complex_non_strict_mode_preserves_partial_bundle_compatibility():
    from relic.computation.operators.dec import DECComplexBundle

    bundle = DECComplexBundle(
        complex_id="partial_compatibility_bundle",
        dimension=3,
        n_cells={0: 1, 1: 1, 2: 1, 3: 1},
        D={0: torch.zeros((1, 1))},
        boundary_masks={},
        orientation={},
        cell_measures={},
        dual_measures={},
        orientation_convention="outward_positive",
    )

    assert bundle.validate_complex(tol=0.0, strict=False) == {}


def test_dec_complex_strict_mode_accepts_complete_consistent_bundle():
    assert _strict_bundle().validate_complex(tol=0.0, strict=True) == {}


@pytest.mark.parametrize("tol", [-1.0, float("nan"), float("inf")])
def test_dec_complex_rejects_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        _strict_bundle().validate_complex(tol=tol, strict=True)


def test_dec_complex_strict_mode_rejects_nonfinite_derivative():
    bundle = _strict_bundle(D={0: torch.tensor([[float("nan"), 1.0]])})

    with pytest.raises(ValueError, match="finite"):
        bundle.validate_complex(strict=True)


def test_dec_complex_strict_mode_rejects_unexpected_degree_keys():
    bundle = _strict_bundle(
        boundary_masks={
            0: torch.tensor([False, True]),
            1: torch.tensor([True]),
            2: torch.tensor([False]),
        }
    )

    with pytest.raises(ValueError, match="unexpected degrees"):
        bundle.validate_complex(strict=True)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {
                "boundary_masks": {
                    0: torch.tensor([0, 2], dtype=torch.uint8),
                    1: torch.tensor([0], dtype=torch.uint8),
                }
            },
            "torch.bool",
        ),
        (
            {
                "orientation": {
                    0: torch.tensor([1, 0], dtype=torch.int8),
                    1: torch.tensor([1], dtype=torch.int8),
                }
            },
            "-1 or 1",
        ),
        (
            {"cell_measures": {0: torch.ones((1, 2)), 1: torch.tensor([1.0])}},
            "rank-1",
        ),
    ],
)
def test_dec_complex_strict_mode_rejects_malformed_support_data(override, message):
    bundle = _strict_bundle(**override)

    with pytest.raises((TypeError, ValueError), match=message):
        bundle.validate_complex(strict=True)


def test_strict_complex_rejects_d0_that_does_not_annihilate_constants():
    bundle = _strict_bundle(
        D={0: torch.tensor([[1.0, 1.0]], dtype=torch.float64)}
    )

    with pytest.raises(ValueError, match="constant"):
        bundle.validate_complex(strict=True)


def test_general_cw_loop_zero_d0_row_remains_valid():
    bundle = _strict_bundle(
        n_cells={0: 1, 1: 1},
        D={0: torch.zeros((1, 1), dtype=torch.float64)},
        boundary_masks={
            0: torch.tensor([False], dtype=torch.bool),
            1: torch.tensor([False], dtype=torch.bool),
        },
        orientation={
            0: torch.tensor([1], dtype=torch.int8),
            1: torch.tensor([1], dtype=torch.int8),
        },
        cell_measures={
            0: torch.ones(1, dtype=torch.float64),
            1: torch.ones(1, dtype=torch.float64),
        },
        dual_measures={
            0: torch.ones(1, dtype=torch.float64),
            1: torch.ones(1, dtype=torch.float64),
        },
    )

    assert bundle.validate_complex(strict=True) == {}


def test_general_integer_cellular_incidence_is_not_limited_to_unit_entries():
    bundle = _strict_triangle_bundle(
        n_cells={0: 1, 1: 1, 2: 1},
        D={
            0: torch.zeros((1, 1), dtype=torch.float64),
            1: torch.tensor([[2.0]], dtype=torch.float64),
        },
        boundary_masks={
            0: torch.tensor([False], dtype=torch.bool),
            1: torch.tensor([False], dtype=torch.bool),
            2: torch.tensor([False], dtype=torch.bool),
        },
        orientation={
            0: torch.tensor([1], dtype=torch.int8),
            1: torch.tensor([1], dtype=torch.int8),
            2: torch.tensor([1], dtype=torch.int8),
        },
        cell_measures={
            0: torch.ones(1, dtype=torch.float64),
            1: torch.ones(1, dtype=torch.float64),
            2: torch.ones(1, dtype=torch.float64),
        },
        dual_measures={
            0: torch.ones(1, dtype=torch.float64),
            1: torch.ones(1, dtype=torch.float64),
            2: torch.ones(1, dtype=torch.float64),
        },
    )

    assert bundle.validate_complex(strict=True)[(1, 0)] == 0.0


def test_strict_complex_rejects_fractional_incidence():
    bundle = _strict_bundle(
        D={0: torch.tensor([[-0.5, 0.5]], dtype=torch.float64)}
    )

    with pytest.raises(ValueError, match="integer-valued"):
        bundle.validate_complex(strict=True)


def test_strict_complex_rejects_incidence_outside_exact_integer_range():
    coefficient = float((2**24) + 2)
    bundle = _strict_bundle(
        D={
            0: torch.tensor(
                [[-coefficient, coefficient]],
                dtype=torch.float32,
            )
        },
        cell_measures={
            0: torch.ones(2, dtype=torch.float32),
            1: torch.ones(1, dtype=torch.float32),
        },
        dual_measures={
            0: torch.ones(2, dtype=torch.float32),
            1: torch.ones(1, dtype=torch.float32),
        },
    )

    with pytest.raises(ValueError, match="exact integer range"):
        bundle.validate_complex(strict=True)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: torch.tensor([[-1, 1]], dtype=torch.int64),
        lambda: torch.tensor([[False, True]], dtype=torch.bool),
        lambda: torch.tensor([[-1.0 + 0.0j, 1.0 + 0.0j]]),
        lambda: torch.tensor([[-1.0, 1.0]], dtype=torch.float16),
        lambda: torch.tensor([[-1.0, 1.0]], dtype=torch.bfloat16),
    ),
    ids=("integer", "bool", "complex", "float16", "bfloat16"),
)
def test_strict_complex_rejects_noncanonical_d_dtype(factory):
    bundle = _strict_bundle(D={0: factory()})

    with pytest.raises(TypeError, match="float32 or float64"):
        bundle.validate_complex(strict=True)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: torch.ones((1, 2), device="meta"),
        lambda: torch.sparse_coo_tensor(
            torch.tensor([[0, 0], [0, 1]]),
            torch.tensor([-1.0, 1.0]),
            (1, 2),
            check_invariants=True,
        ),
        lambda: torch.quantize_per_tensor(
            torch.tensor([[-1.0, 1.0]], dtype=torch.float32),
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        ),
    ),
    ids=("meta", "sparse", "quantized"),
)
@pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor.*:UserWarning")
def test_strict_complex_rejects_unsupported_d_backend(factory):
    bundle = _strict_bundle(D={0: factory()})

    with pytest.raises(TypeError, match="materialized dense strided"):
        bundle.validate_complex(strict=True)


def test_strict_complex_rejects_trainable_topology_operator():
    bundle = _strict_bundle(
        D={
            0: torch.tensor(
                [[-1.0, 1.0]],
                dtype=torch.float64,
                requires_grad=True,
            )
        }
    )

    with pytest.raises(ValueError, match="must not require gradients"):
        bundle.validate_complex(strict=True)


def test_strict_complex_rejects_mixed_operator_dtype_before_composition():
    base = _strict_triangle_bundle()
    bundle = _strict_triangle_bundle(
        D={
            0: base.D[0].to(dtype=torch.float32),
            1: base.D[1].to(dtype=torch.float64),
        }
    )

    with pytest.raises(ValueError, match="one dtype and device"):
        bundle.validate_complex(strict=True)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"dimension": True}, "dimension"),
        ({"D": {False: torch.tensor([[-1.0, 1.0]])}}, "D keys"),
        (
            {
                "n_cells": {0: True, 1: 1},
                "D": {0: torch.zeros((1, 1), dtype=torch.float64)},
            },
            "n_cells",
        ),
    ),
)
def test_complex_identity_rejects_boolean_integer_fields(override, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _strict_bundle(**override)


def test_boundary_matrix_is_the_only_chain_boundary_owner():
    bundle = _strict_bundle()

    boundary = bundle.boundary_matrix(1)

    assert torch.equal(boundary, bundle.D[0].transpose(0, 1))
    assert boundary.data_ptr() == bundle.D[0].data_ptr()
    assert not hasattr(bundle, "boundary_operators")


@pytest.mark.parametrize("degree", (False, 0, 2, "1"))
def test_boundary_matrix_rejects_invalid_degree(degree):
    with pytest.raises((TypeError, ValueError), match="degree"):
        _strict_bundle().boundary_matrix(degree)


def test_orientation_receipt_is_not_applied_twice_to_coboundary():
    bundle = _strict_bundle(
        orientation={
            0: torch.tensor([1, 1], dtype=torch.int8),
            1: torch.tensor([-1], dtype=torch.int8),
        }
    )

    assert torch.equal(bundle.coboundary_matrix(0), bundle.D[0])
    assert torch.equal(bundle.boundary_matrix(1), bundle.D[0].transpose(0, 1))


def test_strict_complex_rejects_float_orientation_receipt():
    bundle = _strict_bundle(
        orientation={
            0: torch.tensor([1.0, 1.0], dtype=torch.float64),
            1: torch.tensor([1.0], dtype=torch.float64),
        }
    )

    with pytest.raises(TypeError, match="torch.int8"):
        bundle.validate_complex(strict=True)


def test_strict_complex_rejects_negative_zero_cell_orientation():
    bundle = _strict_bundle(
        orientation={
            0: torch.tensor([-1, 1], dtype=torch.int8),
            1: torch.tensor([1], dtype=torch.int8),
        }
    )

    with pytest.raises(ValueError, match="degree-0 orientation"):
        bundle.validate_complex(strict=True)


def test_strict_complex_rejects_integer_boundary_mask_even_when_binary():
    bundle = _strict_bundle(
        boundary_masks={
            0: torch.tensor([1, 1], dtype=torch.uint8),
            1: torch.tensor([0], dtype=torch.uint8),
        }
    )

    with pytest.raises(TypeError, match="torch.bool"):
        bundle.validate_complex(strict=True)


def test_strict_complex_rejects_top_degree_boundary_cells():
    bundle = _strict_bundle(
        boundary_masks={
            0: torch.tensor([True, True], dtype=torch.bool),
            1: torch.tensor([True], dtype=torch.bool),
        }
    )

    with pytest.raises(ValueError, match="top-degree"):
        bundle.validate_complex(strict=True)


def test_strict_complex_rejects_boundary_subcomplex_closure_violation():
    bundle = _strict_triangle_bundle(
        boundary_masks={
            0: torch.tensor([True, False, True], dtype=torch.bool),
            1: torch.tensor([True, True, True], dtype=torch.bool),
            2: torch.tensor([False], dtype=torch.bool),
        }
    )

    with pytest.raises(ValueError, match="boundary subcomplex"):
        bundle.validate_complex(strict=True)


def test_strict_triangle_boundary_subcomplex_is_closed():
    assert _strict_triangle_bundle().validate_complex(strict=True)[(1, 0)] == 0.0


@pytest.mark.parametrize("dtype", (torch.bool, torch.int64, torch.complex64, torch.float16))
def test_strict_complex_rejects_noncanonical_measure_dtype(dtype):
    if dtype is torch.bool:
        values = torch.tensor([True, True], dtype=dtype)
    elif dtype.is_complex:
        values = torch.tensor([1.0 + 0.0j, 1.0 + 0.0j], dtype=dtype)
    else:
        values = torch.ones(2, dtype=dtype)
    bundle = _strict_bundle(
        cell_measures={
            0: values,
            1: torch.ones(1, dtype=torch.float64),
        }
    )

    with pytest.raises(TypeError, match="measure.*float32 or float64"):
        bundle.validate_complex(strict=True)


def test_strict_complex_allows_differentiable_positive_measures_without_detach():
    measures = torch.tensor(
        [1.0, 2.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    bundle = _strict_bundle(
        cell_measures={
            0: measures,
            1: torch.ones(1, dtype=torch.float64),
        }
    )

    bundle.validate_complex(strict=True)

    assert bundle.cell_measures[0] is measures
    assert bundle.cell_measures[0].requires_grad is True


def test_strict_complex_rejects_measure_dtype_mismatch():
    bundle = _strict_bundle(
        dual_measures={
            0: torch.ones(2, dtype=torch.float32),
            1: torch.ones(1, dtype=torch.float32),
        }
    )

    with pytest.raises(ValueError, match="measure.*dtype and device"):
        bundle.validate_complex(strict=True)


@pytest.mark.parametrize("tol", (True, "0", torch.tensor(0.0), 0.5))
def test_strict_complex_rejects_noncanonical_or_masking_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        _strict_bundle().validate_complex(tol=tol, strict=True)


def test_declared_boundary_trace_matrix_is_derived_from_masks():
    bundle = _strict_bundle()

    trace0 = bundle.trace_matrix(0)
    trace1 = bundle.trace_matrix(1)

    assert torch.equal(trace0, torch.eye(2, dtype=torch.float64))
    assert trace1.shape == (0, 1)


def test_typed_stokes_audit_holds_and_preserves_input_autograd():
    from relic.computation.operators.dec import TypedChain, TypedCochain

    bundle = _strict_bundle()
    alpha_base = torch.tensor(
        [2.0, 5.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    chain_base = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    alpha_template, chain_template = _typed_interval_pair()
    alpha = TypedCochain(
        name="alpha",
        values=alpha_base.square(),
        space=alpha_template.space,
    )
    chain = TypedChain(
        name="c",
        coefficients=chain_base * 2.0,
        space=chain_template.space,
    )

    audit = bundle.typed_stokes_audit(alpha, chain)

    assert torch.equal(audit["lhs"], audit["rhs"])
    assert audit["residual"].item() == 0.0
    alpha_gradient, chain_gradient = torch.autograd.grad(
        audit["lhs"],
        (alpha_base, chain_base),
    )
    assert torch.equal(alpha_gradient, torch.tensor([-24.0, 60.0]))
    assert torch.equal(chain_gradient, torch.tensor([42.0]))


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    (
        ("alpha", "complex_id", "other", "complex_id"),
        ("alpha", "degree", 1, "degree"),
        ("chain", "primality", "dual", "primality"),
        ("chain", "physical_domain", "other", "physical_domain"),
        (
            "chain",
            "orientation_convention",
            "inward_positive",
            "orientation_convention",
        ),
    ),
)
def test_typed_stokes_audit_rejects_space_mismatch(target, field, value, message):
    from relic.computation.operators.dec import TypedChain, TypedCochain

    alpha, chain = _typed_interval_pair()
    if target == "alpha":
        alpha = TypedCochain(
            name=alpha.name,
            values=alpha.values,
            space=replace(alpha.space, **{field: value}),
        )
    else:
        chain = TypedChain(
            name=chain.name,
            coefficients=chain.coefficients,
            space=replace(chain.space, **{field: value}),
        )

    with pytest.raises(ValueError, match=message):
        _strict_bundle().typed_stokes_audit(alpha, chain)


def test_typed_stokes_audit_rejects_dtype_mismatch():
    alpha, chain = _typed_interval_pair(chain_dtype=torch.float32)

    with pytest.raises(ValueError, match="dtype and device"):
        _strict_bundle().typed_stokes_audit(alpha, chain)
