"""Tests for typed DEC cochain form views."""

from types import SimpleNamespace

import pytest
import torch

from relic.orchestration.ssot_catalog import SSOTEntry


def test_cochain_space_from_ssot_entry_uses_existing_catalog_raw_metadata():
    from relic.computation.operators.dec import CochainSpaceSpec, cochain_space_from_ssot_entry

    entry = SSOTEntry(
        name="soil_water_storage",
        standard_name="soil_water_storage",
        units="kg m-2",
        raw={
            "form_degree": 2,
            "primality": "primal",
            "extensivity": "extensive",
            "physical_domain": "hydrosphere",
            "orientation_convention": "outward_positive",
            "boundary_role": "interior",
        },
    )

    space = cochain_space_from_ssot_entry(
        entry,
        complex_id="soil_column",
        n_dofs=3,
    )

    assert isinstance(space, CochainSpaceSpec)
    assert space.degree == 2
    assert space.primality == "primal"
    assert space.units == "kg m-2"
    assert space.semantic_name == "soil_water_storage"
    assert space.n_dofs == 3


def test_typed_cochain_requires_tensor_payload_and_matching_dofs():
    from relic.computation.operators.dec import CochainSpaceSpec, TypedCochain

    space = CochainSpaceSpec(
        complex_id="soil_column",
        degree=2,
        primality="primal",
        n_dofs=2,
        units="kg m-2",
        extensivity="extensive",
        physical_domain="hydrosphere",
        semantic_name="soil_water_storage",
        orientation_convention="outward_positive",
    )

    cochain = TypedCochain(
        name="soil_water_storage",
        values=torch.tensor([1.0, 2.0]),
        space=space,
    )

    assert cochain.values.shape == (2,)

    with pytest.raises(ValueError, match="n_dofs"):
        TypedCochain(
            name="soil_water_storage",
            values=torch.tensor([1.0, 2.0, 3.0]),
            space=space,
        )

    with pytest.raises(TypeError, match="torch.Tensor"):
        TypedCochain(
            name="soil_water_storage",
            values=[1.0, 2.0],
            space=space,
        )


def test_observation_and_diagnostic_specs_keep_observations_out_of_state():
    from relic.computation.operators.dec import DiagnosticSpec, ObservationOperatorSpec

    observation = ObservationOperatorSpec(
        name="tower_et",
        input_state_vars=("soil_water_storage",),
        output_name="latent_heat_flux_observed",
        output_units="W m-2",
        support="tower_footprint",
        averaging_window="30 min",
        footprint_description="eddy covariance footprint",
        error_model="gaussian",
        conservative_projection_required=True,
    )
    diagnostic = DiagnosticSpec(
        name="water_use_efficiency",
        input_state_vars=("gross_primary_productivity", "evapotranspiration_flux"),
        output_units="g C kg-1 H2O",
        diagnostic_kind="ratio",
    )

    assert observation.conservative_projection_required is True
    assert diagnostic.writes_state is False
    assert diagnostic.feedback_port is None


def test_diagnostic_spec_rejects_state_writes_even_with_feedback_port():
    from relic.computation.operators.dec import DiagnosticSpec

    with pytest.raises(ValueError, match="read-only.*feedback_port cannot authorize"):
        DiagnosticSpec(
            name="forbidden_feedback_diagnostic",
            input_state_vars=("soil_water_storage",),
            output_units="kg m-2",
            diagnostic_kind="feedback",
            writes_state=True,
            feedback_port="soil_water_storage",
        )


@pytest.mark.parametrize(
    "raw",
    [
        {
            "form_degree": 2,
            "extensivity": "extensive",
            "physical_domain": "hydrosphere",
            "orientation_convention": "outward_positive",
        },
        {
            "form_degree": 2,
            "primality": "primal",
            "extensivity": "extensive",
            "physical_domain": "hydrosphere",
        },
        # boundary_role takes part in the strict Hodge boundary-role check, so
        # an undeclared role is not read as "interior".
        {
            "form_degree": 2,
            "primality": "primal",
            "extensivity": "extensive",
            "physical_domain": "hydrosphere",
            "orientation_convention": "outward_positive",
        },
    ],
)
def test_cochain_space_from_ssot_entry_rejects_missing_structural_identity(raw):
    from relic.computation.operators.dec import cochain_space_from_ssot_entry

    entry = SSOTEntry(
        name="soil_water_storage",
        standard_name="soil_water_storage",
        units="kg m-2",
        raw=raw,
    )

    with pytest.raises(ValueError, match="required"):
        cochain_space_from_ssot_entry(entry, complex_id="soil_column", n_dofs=2)


@pytest.mark.parametrize(
    ("degree", "n_dofs", "message"),
    [
        (True, 2, "degree"),
        (1.0, 2, "degree"),
        (1, True, "n_dofs"),
    ],
)
def test_cochain_space_rejects_boolean_or_noninteger_cardinality(degree, n_dofs, message):
    from relic.computation.operators.dec import CochainSpaceSpec

    with pytest.raises(ValueError, match=message):
        CochainSpaceSpec(
            complex_id="soil_column",
            degree=degree,
            primality="primal",
            n_dofs=n_dofs,
            units="kg m-2",
            extensivity="extensive",
            physical_domain="hydrosphere",
            semantic_name="soil_water_storage",
            orientation_convention="outward_positive",
        )


def test_typed_cochain_rejects_implicit_flattening_and_invalid_metadata():
    from relic.computation.operators.dec import CochainSpaceSpec, TypedCochain

    space = CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="primal",
        n_dofs=2,
        units="kg m-2",
        extensivity="extensive",
        physical_domain="hydrosphere",
        semantic_name="soil_water_storage",
        orientation_convention="outward_positive",
    )

    with pytest.raises(ValueError, match="rank-1"):
        TypedCochain(
            name="soil_water_storage",
            values=torch.ones((1, 2)),
            space=space,
        )

    with pytest.raises(ValueError, match="time"):
        TypedCochain(
            name="soil_water_storage",
            values=torch.ones(2),
            space=space,
            time=float("nan"),
        )

    with pytest.raises(ValueError, match="provenance"):
        TypedCochain(
            name="soil_water_storage",
            values=torch.ones(2),
            space=space,
            provenance={"writer": 3},
        )


def _canonical_chain_space(**overrides):
    from relic.computation.operators.dec.forms import ChainSpaceSpec

    payload = {
        "complex_id": "soil_column",
        "degree": 1,
        "primality": "primal",
        "n_dofs": 2,
        "coefficient_units": "1",
        "physical_domain": "hydrosphere",
        "semantic_name": "oriented_edge_chain",
        "orientation_convention": "outward_positive",
        "boundary_role": "interior",
    }
    payload.update(overrides)
    return ChainSpaceSpec(**payload)


def _canonical_cochain_space():
    from relic.computation.operators.dec.forms import CochainSpaceSpec

    return CochainSpaceSpec(
        complex_id="soil_column",
        degree=1,
        primality="primal",
        n_dofs=2,
        units="kg m-2",
        extensivity="extensive",
        physical_domain="hydrosphere",
        semantic_name="soil_water_storage",
        orientation_convention="outward_positive",
    )


def test_typed_cochain_requires_canonical_cochain_space():
    from relic.computation.operators.dec.forms import TypedCochain

    with pytest.raises(TypeError, match="CochainSpaceSpec"):
        TypedCochain(
            name="duck_typed",
            values=torch.tensor([1.0]),
            space=SimpleNamespace(n_dofs=1, units="1"),
        )


def test_chain_and_cochain_runtime_types_are_not_interchangeable():
    from relic.computation.operators.dec.forms import (
        ChainSpaceSpec,
        CochainSpaceSpec,
        TypedChain,
        TypedCochain,
    )

    chain_space = _canonical_chain_space()
    cochain_space = _canonical_cochain_space()
    assert type(chain_space) is ChainSpaceSpec
    assert type(cochain_space) is CochainSpaceSpec
    assert not isinstance(chain_space, CochainSpaceSpec)
    assert not isinstance(cochain_space, ChainSpaceSpec)

    with pytest.raises(TypeError, match="ChainSpaceSpec"):
        TypedChain(
            name="wrong_chain_space",
            coefficients=torch.ones(2),
            space=cochain_space,
        )
    with pytest.raises(TypeError, match="CochainSpaceSpec"):
        TypedCochain(
            name="wrong_cochain_space",
            values=torch.ones(2),
            space=chain_space,
        )


def test_typed_chain_accepts_real_coefficients_and_preserves_autograd():
    from relic.computation.operators.dec.forms import TypedChain

    base = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    coefficients = base.square()
    chain = TypedChain(
        name="integration_chain",
        coefficients=coefficients,
        space=_canonical_chain_space(),
        time=1.5,
        provenance={"source": "unit_test"},
    )

    assert chain.coefficients is coefficients
    assert chain.coefficients.grad_fn is coefficients.grad_fn
    gradient = torch.autograd.grad(chain.coefficients.sum(), base)[0]
    assert torch.equal(gradient, torch.tensor([2.0, 4.0], dtype=torch.float64))


@pytest.mark.parametrize(
    "dtype",
    (torch.bool, torch.int64, torch.complex64, torch.float16, torch.bfloat16),
)
@pytest.mark.parametrize("kind", ("chain", "cochain"))
def test_typed_chain_and_cochain_reject_noncanonical_dtypes(kind, dtype):
    from relic.computation.operators.dec.forms import TypedChain, TypedCochain

    if dtype is torch.bool:
        payload = torch.tensor([True, False], dtype=dtype)
    elif dtype.is_complex:
        payload = torch.tensor([1.0 + 1.0j, 2.0 + 0.0j], dtype=dtype)
    else:
        payload = torch.tensor([1, 2], dtype=dtype)

    with pytest.raises(TypeError, match="float32 or float64"):
        if kind == "chain":
            TypedChain(
                name="invalid_coefficients",
                coefficients=payload,
                space=_canonical_chain_space(),
            )
        else:
            TypedCochain(
                name="invalid_values",
                values=payload,
                space=_canonical_cochain_space(),
            )


@pytest.mark.parametrize("kind", ("chain", "cochain"))
@pytest.mark.parametrize("backend", ("meta", "sparse", "quantized"))
@pytest.mark.filterwarnings("ignore:torch.quantize_per_tensor.*:UserWarning")
def test_typed_chain_and_cochain_reject_unsupported_tensor_backends(kind, backend):
    from relic.computation.operators.dec.forms import TypedChain, TypedCochain

    if backend == "meta":
        payload = torch.ones(2, device="meta")
        message = "materialized"
    elif backend == "sparse":
        payload = torch.sparse_coo_tensor(
            torch.tensor([[0, 1]]),
            torch.tensor([1.0, 2.0]),
            (2,),
            check_invariants=True,
        )
        message = "dense strided"
    else:
        payload = torch.quantize_per_tensor(
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            scale=0.1,
            zero_point=0,
            dtype=torch.qint8,
        )
        message = "dense strided"

    with pytest.raises(TypeError, match=message):
        if kind == "chain":
            TypedChain(
                name="invalid_coefficients",
                coefficients=payload,
                space=_canonical_chain_space(),
            )
        else:
            TypedCochain(
                name="invalid_values",
                values=payload,
                space=_canonical_cochain_space(),
            )


def test_noncontiguous_strided_payloads_remain_valid():
    from relic.computation.operators.dec.forms import TypedChain, TypedCochain

    chain_values = torch.arange(4.0)[::2]
    cochain_values = torch.arange(4.0)[1::2]
    assert not chain_values.is_contiguous()
    assert not cochain_values.is_contiguous()

    chain = TypedChain(
        name="strided_chain",
        coefficients=chain_values,
        space=_canonical_chain_space(),
    )
    cochain = TypedCochain(
        name="strided_cochain",
        values=cochain_values,
        space=_canonical_cochain_space(),
    )

    assert chain.coefficients is chain_values
    assert cochain.values is cochain_values


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("degree", True, "degree"),
        ("n_dofs", 2.0, "n_dofs"),
        ("primality", "cell", "primality"),
        ("complex_id", " ", "complex_id"),
        ("orientation_convention", "", "orientation_convention"),
        ("coefficient_units", "m", "coefficient_units"),
    ),
)
def test_chain_space_rejects_invalid_identity_or_coefficient_units(
    field,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        _canonical_chain_space(**{field: value})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"time": True}, "time"),
        ({"time": float("inf")}, "time"),
        ({"provenance": {"": "source"}}, "provenance"),
        ({"provenance": {"source": " "}}, "provenance"),
    ),
)
def test_typed_chain_rejects_invalid_time_or_provenance(kwargs, message):
    from relic.computation.operators.dec.forms import TypedChain

    with pytest.raises(ValueError, match=message):
        TypedChain(
            name="invalid_metadata",
            coefficients=torch.ones(2),
            space=_canonical_chain_space(),
            **kwargs,
        )


def test_typed_chain_does_not_claim_complex_or_stokes_operators():
    from relic.computation.operators.dec import forms

    assert {"ChainSpaceSpec", "TypedChain"}.issubset(set(forms.__all__))
    chain = forms.TypedChain(
        name="type_only",
        coefficients=torch.ones(2),
        space=_canonical_chain_space(),
    )
    assert not hasattr(chain, "boundary")
    assert not hasattr(chain, "coboundary")
    assert not hasattr(chain, "stokes_residual")


def test_dec_package_exports_the_canonical_chain_types():
    from relic.computation.operators import dec
    from relic.computation.operators.dec.forms import ChainSpaceSpec, TypedChain

    assert dec.ChainSpaceSpec is ChainSpaceSpec
    assert dec.TypedChain is TypedChain
    assert {"ChainSpaceSpec", "TypedChain"}.issubset(set(dec.__all__))
