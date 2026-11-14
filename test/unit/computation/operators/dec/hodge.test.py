"""Tests for DEC Hodge star specifications."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def _spaces():
    from relic.computation.operators.dec import CochainSpaceSpec

    domain = CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="primal",
        n_dofs=2,
        units="m",
        extensivity="integral",
        physical_domain="hydrosphere",
        semantic_name="hydraulic_head_difference",
        orientation_convention="outward_positive",
    )
    codomain = CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="dual",
        n_dofs=2,
        units="kg s-1",
        extensivity="integral",
        physical_domain="hydrosphere",
        semantic_name="water_flux",
        orientation_convention="outward_positive",
    )
    return domain, codomain


def test_hodge_star_accepts_positive_diagonal_passive_operator():
    from relic.computation.operators.dec import HodgeStarSpec

    domain, codomain = _spaces()
    spec = HodgeStarSpec(
        name="soil_water_hodge",
        domain=domain,
        codomain=codomain,
        matrix=torch.diag(torch.tensor([1.0, 2.0])),
        parameter_keys=("hydrosphere.soil.transport.hodge.hydraulic_conductivity",),
        metric_keys=("hydrosphere.soil.transport.metric.face_area",),
        material_law="darcy",
    )

    audit = spec.validate(passive=True)

    assert audit["shape_ok"] is True
    assert audit["symmetric"] is True
    assert audit["psd"] is True
    assert audit["min_eigenvalue"] == pytest.approx(1.0)


def test_hodge_star_rejects_negative_diagonal():
    from relic.computation.operators.dec import HodgeStarSpec

    domain, codomain = _spaces()
    spec = HodgeStarSpec(
        name="bad_hodge",
        domain=domain,
        codomain=codomain,
        matrix=torch.diag(torch.tensor([1.0, -0.5])),
        parameter_keys=("k_bad",),
        metric_keys=("area",),
        material_law="darcy",
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        spec.validate(passive=True)


def test_hodge_star_rejects_nonsymmetric_square_passive_operator():
    from relic.computation.operators.dec import HodgeStarSpec

    domain, codomain = _spaces()
    spec = HodgeStarSpec(
        name="nonsymmetric_hodge",
        domain=domain,
        codomain=codomain,
        matrix=torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
        parameter_keys=("k",),
        metric_keys=("area",),
        material_law="darcy",
    )

    with pytest.raises(ValueError, match="symmetric"):
        spec.validate(passive=True)


def test_hodge_star_rejects_shape_mismatch():
    from relic.computation.operators.dec import HodgeStarSpec

    domain, codomain = _spaces()

    with pytest.raises(ValueError, match="shape"):
        HodgeStarSpec(
            name="shape_bad_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.ones(3, 2),
            parameter_keys=("k",),
            metric_keys=("area",),
            material_law="darcy",
        )


def _strict_spec(**overrides):
    from relic.computation.operators.dec import HodgeStarSpec

    domain, codomain = _spaces()
    values = {
        "name": "strict_soil_water_hodge",
        "domain": domain,
        "codomain": codomain,
        "matrix": torch.eye(2),
        "parameter_keys": ("hydrosphere.soil.transport.hodge.k",),
        "metric_keys": ("hydrosphere.soil.transport.metric.area",),
        "material_law": "darcy",
        "complex_dimension": 2,
        "operator_units": "kg m-1 s-1",
        "role": "metric_hodge",
        "definiteness": "positive_definite",
    }
    values.update(overrides)
    return HodgeStarSpec(**values)


def test_hodge_star_strict_mode_accepts_complete_bound_operator():
    audit = _strict_spec().validate(
        passive=True,
        strict=True,
        parameter_resolver=lambda key: {"key": key},
    )

    assert audit["structural_duality"] is True
    assert audit["parameter_binding"] is True
    assert audit["unit_relation"] is True
    assert audit["definiteness_verified"] is True


def test_hodge_star_strict_passive_mode_rejects_rectangular_operator():
    domain, codomain = _spaces()
    rectangular_codomain = replace(codomain, n_dofs=3)
    spec = _strict_spec(
        domain=domain,
        codomain=rectangular_codomain,
        matrix=torch.ones((3, 2)),
    )

    with pytest.raises(ValueError, match="square"):
        spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})


@pytest.mark.parametrize(
    ("codomain_update", "message"),
    [
        ({"complex_id": "other_complex"}, "same complex"),
        ({"primality": "primal"}, "opposite primality"),
        ({"orientation_convention": "inward_positive"}, "orientation"),
        ({"degree": 0}, "complementary"),
    ],
)
def test_hodge_star_strict_mode_rejects_invalid_structural_duality(
    codomain_update,
    message,
):
    domain, codomain = _spaces()
    spec = _strict_spec(codomain=replace(codomain, **codomain_update))

    with pytest.raises(ValueError, match=message):
        spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})


def test_hodge_star_strict_mode_rejects_unresolved_parameter_key():
    with pytest.raises(ValueError, match="ParameterManager key"):
        _strict_spec().validate(strict=True, parameter_resolver=lambda key: None)


def test_hodge_star_strict_mode_rejects_inconsistent_operator_units():
    with pytest.raises(ValueError, match="unit relation"):
        _strict_spec(operator_units="s-1").validate(
            strict=True,
            parameter_resolver=lambda key: {"key": key},
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"symmetry_tol": float("nan")},
        {"psd_tol": -1.0},
        {"matrix": torch.tensor([[1.0, float("inf")], [0.0, 1.0]])},
    ],
)
def test_hodge_star_rejects_invalid_numeric_contract(overrides):
    with pytest.raises(ValueError):
        _strict_spec(**overrides)


@pytest.mark.parametrize(
    ("field_name", "invalid_space"),
    [
        (
            "domain",
            SimpleNamespace(
                complex_id="soil_column",
                degree=1,
                primality="primal",
                n_dofs=2,
                units="m",
            ),
        ),
        (
            "codomain",
            SimpleNamespace(
                complex_id="soil_column",
                degree=1,
                primality="dual",
                n_dofs=2,
                units="kg s-1",
            ),
        ),
    ],
)
def test_hodge_star_rejects_duck_typed_space(field_name, invalid_space):
    with pytest.raises(TypeError, match="CochainSpaceSpec"):
        _strict_spec(**{field_name: invalid_space})


def test_hodge_star_rejects_chain_space_as_domain():
    from relic.computation.operators.dec import ChainSpaceSpec

    chain_space = ChainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="primal",
        n_dofs=2,
        coefficient_units="1",
        physical_domain="hydrosphere",
        semantic_name="edge_chain",
        orientation_convention="outward_positive",
    )

    with pytest.raises(TypeError, match="CochainSpaceSpec"):
        _strict_spec(domain=chain_space)


@pytest.mark.parametrize(
    "matrix",
    [
        torch.ones((2, 2), dtype=torch.bool),
        torch.ones((2, 2), dtype=torch.int64),
        torch.ones((2, 2), dtype=torch.complex64),
        torch.ones((2, 2), dtype=torch.float16),
        torch.ones((2, 2), dtype=torch.bfloat16),
        torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]]),
            torch.ones(2),
            (2, 2),
            check_invariants=False,
        ),
        torch.empty((2, 2), device="meta"),
        torch.quantize_per_tensor(
            torch.ones((2, 2), dtype=torch.float32),
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        ),
    ],
)
def test_hodge_star_rejects_invalid_matrix_tensor_domain(matrix):
    with pytest.raises(TypeError, match="matrix"):
        _strict_spec(matrix=matrix)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hodge_star_preserves_noncontiguous_matrix_identity_and_autograd(dtype):
    base = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=dtype, requires_grad=True)
    matrix = base.transpose(0, 1)
    assert matrix.is_contiguous() is False

    spec = _strict_spec(matrix=matrix)
    audit = spec.validate(
        strict=True,
        parameter_resolver=lambda key: {"key": key},
    )

    assert spec.matrix is matrix
    assert spec.matrix.grad_fn is matrix.grad_fn
    assert audit["definiteness_verified"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"role": "operator"},
        {"definiteness": "nonnegative"},
    ],
)
def test_hodge_star_rejects_invalid_role_or_definiteness(overrides):
    with pytest.raises(ValueError):
        _strict_spec(**overrides)


def test_hodge_star_strict_mode_rejects_unspecified_role():
    with pytest.raises(ValueError, match="role"):
        _strict_spec(role="unspecified").validate(
            strict=True,
            parameter_resolver=lambda key: {"key": key},
        )


def test_hodge_star_strict_mode_rejects_unspecified_definiteness():
    with pytest.raises(ValueError, match="definiteness"):
        _strict_spec(definiteness="unspecified").validate(
            strict=True,
            parameter_resolver=lambda key: {"key": key},
        )


def test_metric_hodge_rejects_positive_semidefinite_declaration():
    with pytest.raises(ValueError, match="metric_hodge"):
        _strict_spec(definiteness="positive_semidefinite").validate(
            strict=True,
            parameter_resolver=lambda key: {"key": key},
        )


@pytest.mark.parametrize(
    "matrix",
    [
        torch.zeros((2, 2), dtype=torch.float64),
        torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)),
    ],
)
def test_metric_hodge_rejects_zero_or_singular_matrix(matrix):
    with pytest.raises(ValueError, match="positive definite"):
        _strict_spec(matrix=matrix).validate(
            strict=True,
            parameter_resolver=lambda key: {"key": key},
        )


def test_metric_hodge_rejects_kernel_basis():
    with pytest.raises(ValueError, match="kernel"):
        _strict_spec(kernel_basis=torch.empty((2, 0))).validate(
            strict=True,
            parameter_resolver=lambda key: {"key": key},
        )


def test_material_positive_definite_hodge_requires_no_kernel():
    spec = _strict_spec(
        role="material_hodge",
        definiteness="positive_definite",
    )
    audit = spec.validate(
        strict=True,
        parameter_resolver=lambda key: {"key": key},
    )

    assert audit["definiteness_verified"] is True
    assert audit["kernel_dimension"] is None


def test_material_positive_definite_hodge_rejects_empty_kernel_tensor():
    spec = _strict_spec(
        role="material_hodge",
        definiteness="positive_definite",
        kernel_basis=torch.empty((2, 0)),
    )

    with pytest.raises(ValueError, match="kernel"):
        spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})


@pytest.mark.parametrize(
    "kernel_basis",
    [None, torch.empty((2, 0), dtype=torch.float64)],
)
def test_material_positive_semidefinite_hodge_requires_nonempty_kernel(kernel_basis):
    spec = _strict_spec(
        role="material_hodge",
        definiteness="positive_semidefinite",
        matrix=torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64)),
        kernel_basis=kernel_basis,
    )

    with pytest.raises(ValueError, match="kernel"):
        spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})


def test_material_positive_semidefinite_hodge_accepts_complete_kernel():
    spec = _strict_spec(
        role="material_hodge",
        definiteness="positive_semidefinite",
        matrix=torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64)),
        kernel_basis=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
    )

    audit = spec.validate(
        strict=True,
        parameter_resolver=lambda key: {"key": key},
    )

    assert audit["definiteness_verified"] is True
    assert audit["kernel_dimension"] == 1
    assert audit["active_rank"] == 1
    assert audit["kernel_residual"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("kernel_basis", "message"),
    [
        (torch.ones((3, 1), dtype=torch.float64), "shape"),
        (torch.ones((2, 1), dtype=torch.float32), "dtype"),
        (torch.ones((2, 1), dtype=torch.float64, requires_grad=True), "requires_grad"),
    ],
)
def test_hodge_star_rejects_invalid_kernel_tensor_contract(kernel_basis, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _strict_spec(
            role="material_hodge",
            definiteness="positive_semidefinite",
            matrix=torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64)),
            kernel_basis=kernel_basis,
        )


@pytest.mark.parametrize(
    ("matrix", "kernel_basis", "message"),
    [
        (
            torch.diag(torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)),
            torch.tensor(
                [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
                dtype=torch.float64,
            ),
            "independent",
        ),
        (
            torch.diag(torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)),
            torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64),
            "nullity",
        ),
        (
            torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float64)),
            torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            "residual",
        ),
    ],
)
def test_material_kernel_must_be_independent_complete_and_annihilated(
    matrix,
    kernel_basis,
    message,
):
    domain, codomain = _spaces()
    if matrix.shape[0] != 2:
        domain = replace(domain, n_dofs=matrix.shape[0])
        codomain = replace(codomain, n_dofs=matrix.shape[0])
    spec = _strict_spec(
        role="material_hodge",
        definiteness="positive_semidefinite",
        domain=domain,
        codomain=codomain,
        matrix=matrix,
        kernel_basis=kernel_basis,
    )

    with pytest.raises(ValueError, match=message):
        spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})


def test_zero_material_hodge_accepts_only_full_kernel_basis():
    full_kernel = _strict_spec(
        role="material_hodge",
        definiteness="positive_semidefinite",
        matrix=torch.zeros((2, 2), dtype=torch.float64),
        kernel_basis=torch.eye(2, dtype=torch.float64),
    )
    audit = full_kernel.validate(
        strict=True,
        parameter_resolver=lambda key: {"key": key},
    )

    assert audit["zero_operator"] is True
    assert audit["active_rank"] == 0
    assert audit["kernel_dimension"] == 2

    incomplete_kernel = replace(
        full_kernel,
        kernel_basis=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="nullity"):
        incomplete_kernel.validate(
            strict=True,
            parameter_resolver=lambda key: {"key": key},
        )


def test_rectangular_passive_hodge_is_rejected_even_when_nonstrict():
    domain, codomain = _spaces()
    spec = _strict_spec(
        role="unspecified",
        definiteness="unspecified",
        codomain=replace(codomain, n_dofs=3),
        matrix=torch.ones((3, 2)),
    )

    with pytest.raises(ValueError, match="square"):
        spec.validate(passive=True, strict=False)


def test_rectangular_nonpassive_operator_has_no_false_definiteness_claim():
    domain, codomain = _spaces()
    spec = _strict_spec(
        role="unspecified",
        definiteness="unspecified",
        codomain=replace(codomain, n_dofs=3),
        matrix=torch.ones((3, 2)),
    )

    audit = spec.validate(passive=False, strict=False)

    assert audit["passivity_evaluated"] is False
    assert audit["symmetric"] is False
    assert audit["psd"] is False
    assert audit["min_eigenvalue"] is None
    assert audit["definiteness_verified"] is False
    assert all(not isinstance(value, float) or not torch.isnan(torch.tensor(value)) for value in audit.values())


def test_strict_hodge_cannot_disable_passivity():
    with pytest.raises(ValueError, match="passive"):
        _strict_spec().validate(
            passive=False,
            strict=True,
            parameter_resolver=lambda key: {"key": key},
        )


def test_nonstrict_zero_hodge_reports_numeric_psd_without_verification():
    spec = _strict_spec(
        role="unspecified",
        definiteness="unspecified",
        matrix=torch.zeros((2, 2), dtype=torch.float64),
    )

    audit = spec.validate(passive=True, strict=False)

    assert audit["symmetric"] is True
    assert audit["psd"] is True
    assert audit["min_eigenvalue"] == pytest.approx(0.0)
    assert audit["definiteness_verified"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"passive": "false"},
        {"strict": 1},
    ],
)
def test_hodge_star_validate_rejects_truthy_nonbool_flags(kwargs):
    with pytest.raises(TypeError, match="bool"):
        _strict_spec().validate(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"symmetry_tol": True},
        {"symmetry_tol": 0.5},
        {"psd_tol": float("inf")},
        {"psd_tol": 0.5},
        {"kernel_rtol": True},
        {"kernel_rtol": 0.5},
    ],
)
def test_hodge_star_rejects_invalid_relative_tolerance(overrides):
    with pytest.raises(ValueError, match="tolerance"):
        _strict_spec(**overrides)


def test_strict_hodge_rejects_configured_tolerance_above_dtype_precision_gate():
    spec = _strict_spec(
        matrix=torch.eye(2, dtype=torch.float64),
        psd_tol=1.0e-3,
    )

    with pytest.raises(ValueError, match="strict tolerance"):
        spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})


def test_positive_definite_eigenvalue_equal_to_threshold_fails():
    spec = _strict_spec(
        matrix=torch.diag(torch.tensor([1.0e-10, 1.0], dtype=torch.float64)),
        psd_tol=1.0e-10,
    )

    with pytest.raises(ValueError, match="positive definite"):
        spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})


def test_material_psd_relative_threshold_and_kernel_residual_are_scale_aware():
    for scale in (1.0, 1.0e8):
        matrix = torch.diag(
            torch.tensor([-5.0e-11 * scale, 1.0 * scale], dtype=torch.float64)
        )
        spec = _strict_spec(
            role="material_hodge",
            definiteness="positive_semidefinite",
            matrix=matrix,
            kernel_basis=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
            psd_tol=1.0e-10,
        )
        audit = spec.validate(
            strict=True,
            parameter_resolver=lambda key: {"key": key},
        )
        assert audit["definiteness_verified"] is True


def test_material_psd_rejects_negative_eigenvalue_beyond_relative_threshold():
    spec = _strict_spec(
        role="material_hodge",
        definiteness="positive_semidefinite",
        matrix=torch.diag(torch.tensor([-2.0e-10, 1.0], dtype=torch.float64)),
        kernel_basis=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        psd_tol=1.0e-10,
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})


def test_trainable_matrix_validation_is_pointwise_and_rechecks_mutation():
    matrix = torch.eye(2, dtype=torch.float64, requires_grad=True)
    spec = _strict_spec(matrix=matrix)
    first = spec.validate(
        strict=True,
        parameter_resolver=lambda key: {"key": key},
    )
    assert first["pointwise_current_value_only"] is True

    with torch.no_grad():
        matrix[1, 1] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})


# MA-21: the default relative PSD tolerance tau_psd = psd_tol * max|lambda| is read
# from ParameterManager key validation.iphs.hodge_psd_rtol; an explicit psd_tol
# overrides it. Test matrix diag(-5e-11, 1): lambda_min = -5e-11, max|lambda| = 1.
_HODGE_PSD_RTOL_KEY = "validation.iphs.hodge_psd_rtol"


def _slightly_indefinite_passive_spec(**overrides):
    from relic.computation.operators.dec import HodgeStarSpec

    domain, codomain = _spaces()
    fields = dict(
        name="slightly_indefinite_hodge",
        domain=domain,
        codomain=codomain,
        matrix=torch.diag(torch.tensor([-5.0e-11, 1.0], dtype=torch.float64)),
        parameter_keys=("hydrosphere.soil.transport.hodge.hydraulic_conductivity",),
        metric_keys=("hydrosphere.soil.transport.metric.face_area",),
        material_law="darcy",
    )
    fields.update(overrides)
    return HodgeStarSpec(**fields)


def test_hodge_psd_tolerance_default_resolves_from_the_parameter_database():
    from pathlib import Path

    import yaml

    from relic.parameters import get_global_manager

    path = (
        Path(__file__).resolve().parents[5]
        / "relic" / "parameters" / "database" / "relic" / "validation" / "iphs.yaml"
    )
    entry = yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][_HODGE_PSD_RTOL_KEY]
    for metadata_field in ("value", "units", "dtype", "bounds", "category", "description",
                           "references", "tags", "requires_grad", "provenance"):
        assert metadata_field in entry, metadata_field
    value = float(entry["value"])
    assert get_global_manager().get(_HODGE_PSD_RTOL_KEY, required=True) == pytest.approx(value)
    spec = _slightly_indefinite_passive_spec()
    assert spec.psd_tol == pytest.approx(value)
    assert spec.validate(passive=True)["configured_psd_rtol"] == pytest.approx(value)
    assert _slightly_indefinite_passive_spec(psd_tol=1.0e-9).psd_tol == pytest.approx(1.0e-9)


def test_changing_the_hodge_psd_parameter_changes_the_passivity_check():
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    original = manager.get(_HODGE_PSD_RTOL_KEY, required=True)
    assert _slightly_indefinite_passive_spec().validate(passive=True)["psd"] is True
    try:
        manager.set_value(_HODGE_PSD_RTOL_KEY, 1.0e-11)
        with pytest.raises(ValueError, match="positive semidefinite"):
            _slightly_indefinite_passive_spec().validate(passive=True)
    finally:
        manager.set_value(_HODGE_PSD_RTOL_KEY, original)
    assert manager.get(_HODGE_PSD_RTOL_KEY, required=True) == original


# MA-21: symmetry_tol (rho_sym = ||H - H^T||_F / ||H||_F) and kernel_rtol (normalized
# ||H K||_F / (||H||_2 ||K||_F) and the kernel rank cut) default to ParameterManager keys.
_HODGE_RELATIVE_TOLERANCE_KEYS = {
    "symmetry_tol": "validation.iphs.hodge_symmetry_rtol",
    "kernel_rtol": "validation.iphs.hodge_kernel_rtol",
}
_YAML_METADATA_FIELDS = ("value", "units", "dtype", "bounds", "category", "description",
                         "references", "tags", "requires_grad", "provenance")


def _validation_iphs_entry(key):
    from pathlib import Path

    import yaml

    path = (
        Path(__file__).resolve().parents[5]
        / "relic" / "parameters" / "database" / "relic" / "validation" / "iphs.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))["parameters"][key]


@pytest.mark.parametrize("field_name,key", sorted(_HODGE_RELATIVE_TOLERANCE_KEYS.items()))
def test_hodge_relative_tolerance_defaults_resolve_from_the_parameter_database(field_name, key):
    from relic.parameters import get_global_manager

    entry = _validation_iphs_entry(key)
    for metadata_field in _YAML_METADATA_FIELDS:
        assert metadata_field in entry, metadata_field
    value = float(entry["value"])
    assert get_global_manager().get(key, required=True) == pytest.approx(value)
    assert getattr(_slightly_indefinite_passive_spec(), field_name) == pytest.approx(value)
    assert getattr(
        _slightly_indefinite_passive_spec(**{field_name: 1.0e-9}), field_name
    ) == pytest.approx(1.0e-9)


def _nearly_symmetric_check(tolerance_changed):
    # rho_sym = sqrt(2) 1e-11 / sqrt(2) = 1e-11.
    spec = _slightly_indefinite_passive_spec(
        matrix=torch.tensor([[1.0, 1.0e-11], [0.0, 1.0]], dtype=torch.float64)
    )
    if tolerance_changed:
        with pytest.raises(ValueError, match="not symmetric"):
            spec.validate(passive=True)
    else:
        assert spec.validate(passive=True)["symmetric"] is True


def _kernel_residual_check(tolerance_changed):
    # H = diag(0, 1), K = (1, 5e-11)^T: ||H K||_F / (||H||_2 ||K||_F) = 5e-11.
    spec = _strict_spec(
        role="material_hodge",
        definiteness="positive_semidefinite",
        matrix=torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64)),
        kernel_basis=torch.tensor([[1.0], [5.0e-11]], dtype=torch.float64),
    )
    if tolerance_changed:
        with pytest.raises(ValueError, match="kernel_basis residual"):
            spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})
    else:
        audit = spec.validate(strict=True, parameter_resolver=lambda key: {"key": key})
        assert audit["kernel_residual"] == pytest.approx(5.0e-11, rel=1.0e-6)


@pytest.mark.parametrize("field_name,check", [
    ("symmetry_tol", _nearly_symmetric_check), ("kernel_rtol", _kernel_residual_check),
])
def test_changing_a_hodge_relative_tolerance_parameter_changes_the_check(field_name, check):
    from relic.parameters import get_global_manager

    key = _HODGE_RELATIVE_TOLERANCE_KEYS[field_name]
    manager = get_global_manager()
    original = manager.get(key, required=True)
    check(tolerance_changed=False)
    try:
        manager.set_value(key, 1.0e-12)
        check(tolerance_changed=True)
    finally:
        manager.set_value(key, original)
    assert manager.get(key, required=True) == original
